# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""FlyDSL native dispatch split into dispatch_send / dispatch_recv, on one node.

A torchrun script, not a pytest file -- it needs eight ranks and a cco
communicator, like its neighbours here.

    torchrun --standalone --nproc_per_node=8 test_native_split.py

Env: MODES (comma list of _MODES keys; default: every mode this arch runs), LOOP
(iterations per mode, at least 3), STRESS_ROUNDS (default 1000; 0 skips),
GRAPH_REPLAYS (default 32; 0 skips), PERF_ITERS (default 100; 0 skips).

The split must be the fused kernel cut in two, nothing more, so most checks
compare the two APIs on the same op and the same inputs:

  * **split == fused** -- received tokens, weights, indices and scales, keyed by
    their origin (rank, token) because slot order comes from atomics and differs
    run to run; the receive count; which (token, k) entries the forward map
    (tok_map) sends to which PE, and which it drops as duplicates; the combine
    output, bit for bit. Both must also match the source rows.
  * **replay** -- a replay dispatch reuses the slots, so there the split and
    fused outputs are compared position by position, with the arena zeroed in
    between so a split that wrote nothing cannot pass.
  * **capped receive** -- with max_total_recv_tokens below the worst case the
    sender drops what does not fit; the count must be exactly the cap.

Then the protocol, which a per-phase sync never exercises:

  * **stress** -- STRESS_ROUNDS rounds with no host sync; split and fused
    alternate, then each rank picks its API independently (the two speak the
    same protocol), and random ranks sleep or run a GEMM between send and recv.
    Every round's count and combine output are checked on the device; at the
    end the counters the recv half clears (recv_num, tok_off, dest_pe_counter)
    must be zero on every rank.
  * **call order** -- a second send, a dispatch, a combine or a
    local_expert_count between send and recv is refused on the host before
    launching, and the op is usable after finishing the round or after reset().
  * **graph** -- send -> GEMM -> recv -> combine captured once and replayed with
    fresh payloads, a zero-input rank included.

Perf (rank 0 prints, no pass/fail): fused vs split per token count, and fused +
GEMM vs send + GEMM + recv, which is the overlap the split exists for.
"""

import os
import sys

import torch
import torch.distributed as dist
from mori.cco.communicator import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp
from mori.tensor_utils import from_gpu_ptr


def _arch():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _fp8():
    return torch.float8_e4m3fn if _arch().startswith("gfx95") else torch.float8_e4m3fnuz


_FP4 = torch.float4_e2m1fn_x2

# name -> (hidden, topk, experts_per_rank, max_tokens, dispatch, scale (block,
# bytes per scale) or None, extra config). The dispatch dtypes and scale layouts
# are the ones SGLang's MORI EPv2 uses: fp8 with fp32 scales per 128, MXFP8 and
# MXFP4 with e8m0 scales per 32, both with a bf16 combine.
_MODES = {
    "bf16": (7168, 8, 32, 128, "bf16", None, {}),
    "bf16_topk6": (4096, 6, 48, 96, "bf16", None, {}),
    "fp8_blockwise_scales": (7168, 8, 32, 128, "fp8", (128, 4), {}),
    "mxfp8": (7168, 6, 48, 128, "fp8", (32, 1), {}),
    "mxfp4": (7168, 6, 48, 128, "fp4", (32, 1), {}),
    "bf16_scatter_fp8": (
        4096,
        8,
        16,
        64,
        "bf16",
        None,
        {"combine_mode": "scatter", "quant_type": "fp8_blockwise"},
    ),
    # A quarter of the worst case: most rounds overflow.
    "bf16_capped": (2048, 8, 8, 64, "bf16", None, {"max_total_recv_tokens": 128}),
}


def _default_modes():
    modes = list(_MODES)
    if not _arch().startswith("gfx95"):
        modes.remove("mxfp4")  # fp4 is gfx950-only
    return modes


class Dist:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    def bcast_uid(self, uid):
        objs = [uid if self.rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        return objs[0]

    def allreduce_sum(self, value):
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t)
        return int(t.item())

    def allreduce_max(self, value):
        t = torch.tensor([value], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())


def lockstep():
    torch.cuda.synchronize()
    dist.barrier()


def _config(d, hidden, topk, epr, m, disp, scale, extra):
    kw = dict(extra)
    if disp != "bf16":
        kw.update(
            dispatch_data_type=_FP4 if disp == "fp4" else _fp8(),
            combine_data_type=torch.bfloat16,
        )
    if scale is not None:
        kw.update(scale_dim=hidden // scale[0], scale_type_size=scale[1])
    return EpDispatchCombineConfig(
        rank=d.rank,
        world_size=d.world,
        hidden_dim=hidden,
        max_num_inp_token_per_rank=m,
        num_experts_per_rank=epr,
        num_experts_per_token=topk,
        data_type=torch.bfloat16,
        kernel_backend="flydsl",
        **kw,
    )


def _payload(disp, rows, hidden, gen, dev):
    """Tokens in the transport dtype. Dispatch only moves bytes, so fp8/fp4 are
    random bytes; bf16 is random values, for the combine round trip."""
    if disp == "bf16":
        return torch.randn(rows, hidden, generator=gen, device=dev).to(torch.bfloat16)
    cols = hidden // 2 if disp == "fp4" else hidden
    raw = torch.randint(0, 256, (rows, cols), generator=gen, device=dev)
    return raw.to(torch.uint8).view(_FP4 if disp == "fp4" else _fp8())


def _scales(scale, rows, hidden, gen, dev):
    if scale is None:
        return None
    nbytes = (hidden // scale[0]) * scale[1]
    assert nbytes % 4 == 0, "the kernel moves scale rows as whole dwords"
    raw = torch.randint(0, 256, (rows, nbytes), generator=gen, device=dev)
    return raw.to(torch.uint8)


def _routing(world, m, topk, epr, gen, dev):
    total = epr * world
    return torch.stack(
        [torch.randperm(total, generator=gen, device=dev)[:topk] for _ in range(m)]
    ).to(torch.int32)


def _bytes(t):
    """Row-wise raw bytes of a 2-D tensor, so every dtype (fp4 included) compares
    the same way."""
    return t.contiguous().view(torch.uint8)


def _origin_table(world, m, hidden, dev):
    """Combine input per origin (rank * m + token): small integers, so the
    identity-expert sum is exact in bf16 and the same on every rank."""
    o = torch.arange(world * m, device=dev).view(-1, 1)
    c = torch.arange(hidden, device=dev).view(1, -1)
    return (((o * 7 + c) % 13) - 6).to(torch.bfloat16)


def _distinct_pes(idx_cpu, epr):
    return torch.tensor(
        [len({int(e) // epr for e in row.tolist()}) for row in idx_cpu],
        dtype=torch.float32,
    ).view(-1, 1)


def _gather(n, idx, x, w, s, world):
    local = (
        n,
        idx.cpu(),
        _bytes(x).cpu(),
        w.cpu(),
        None if s is None else s.cpu(),
    )
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    return gathered


def _expected_count(rank, gathered, epr):
    return sum(
        rank in {int(e) // epr for e in row.tolist()}
        for _, idx, _, _, _ in gathered
        for row in idx
    )


class _Snapshot:
    """Host copies of one dispatch's results, taken after lockstep()."""

    def __init__(self, op, ret, n, topk):
        out, weights, scales, indices, total, routing = ret
        self.recv = int(total[0].item())
        self.reverse = routing.disp_tok_id_to_src_tok_id_local[: self.recv].cpu()
        self.rows = {
            "tokens": _bytes(out[: self.recv]).cpu(),
            "weights": weights[: self.recv].cpu(),
            "indices": indices[: self.recv].cpu(),
        }
        if scales is not None:
            self.rows["scales"] = scales[: self.recv].cpu()
        cap = op.cfg.effective_max_recv
        # Destination PE per (token, k); npes marks a duplicate or overflow.
        self.dest_pe = (routing.disp_dest_tok_id_map[: n * topk].cpu() // cap).view(
            n, topk
        )
        self.routing = routing

    def by_origin(self):
        order = torch.argsort(self.reverse.to(torch.int64))
        return self.reverse[order], {k: v[order] for k, v in self.rows.items()}


def _check_against_source(d, snap, gathered, m, tag):
    """Every received row is its origin's source row, byte for byte."""
    bad = 0
    origins = snap.reverse.to(torch.int64)
    pes, toks = origins // m, origins % m
    valid = torch.tensor(
        [
            0 <= p < d.world and 0 <= t < gathered[p][0]
            for p, t in zip(pes.tolist(), toks.tolist())
        ],
        dtype=torch.bool,
    )
    if not bool(valid.all()):
        print(f"[rank {d.rank}] {tag}: {(~valid).sum()} invalid origins", flush=True)
        return 1
    if len(set(origins.tolist())) != len(origins):
        print(f"[rank {d.rank}] {tag}: an origin arrived twice", flush=True)
        bad += 1
    fields = {"tokens": 2, "indices": 1, "weights": 3, "scales": 4}
    for pe in range(d.world):
        sel = pes == pe
        if not bool(sel.any()):
            continue
        for name, slot in fields.items():
            if name not in snap.rows:
                continue
            want = gathered[pe][slot][toks[sel]]
            got = snap.rows[name][sel]
            if name == "scales":
                got = _bytes(got)
            if not torch.equal(got, want):
                print(
                    f"[rank {d.rank}] {tag}: {name} from rank {pe} differ", flush=True
                )
                bad += 1
    return bad


def _combine(op, snap, table, dev):
    stage = table[snap.reverse.to(dev).to(torch.int64)]
    out, _ = op.combine(stage, None, None, routing=snap.routing)
    return out


def run_mode(d, comm, name, loops):
    hidden, topk, epr, m, disp, scale, extra = _MODES[name]
    rank, world, dev = d.rank, d.world, d.dev
    cfg = _config(d, hidden, topk, epr, m, disp, scale, extra)
    table = _origin_table(world, m, hidden, dev)
    quantized = extra.get("quant_type", "none") != "none"
    capped = "max_total_recv_tokens" in extra
    failures = 0
    with EpDispatchCombineOp(cfg, comm) as op:
        if "split_dispatch" not in op.capabilities:
            print(f"[rank {rank}] {name}: no split_dispatch capability", flush=True)
            return 1
        cap = cfg.effective_max_recv
        for it in range(loops):
            # Uneven per-rank counts, including zero and the full m.
            n = [0, m, 1, m // 3, m - 1, 7, m // 2, 5][(rank + it) % 8]
            gen = torch.Generator(device=dev).manual_seed(97 + rank * 13 + it)
            x = _payload(disp, n, hidden, gen, dev)
            w = torch.rand(n, topk, generator=gen, device=dev)
            s = _scales(scale, n, hidden, gen, dev)
            idx = _routing(world, m, topk, epr, gen, dev)[:n]
            gathered = _gather(n, idx, x, w, s, world)
            want = _expected_count(rank, gathered, epr)
            if capped:
                want = min(want, cap)
            tag = f"{name} it={it}"

            snaps, combined = {}, {}
            for api in ("fused", "split"):
                if api == "fused":
                    ret = op.dispatch(x, w, s, idx, return_routing=True)
                else:
                    ret = op.dispatch_send(x, w, s, idx, return_routing=True)
                    op.dispatch_recv()
                lockstep()
                snap = snaps[api] = _Snapshot(op, ret, n, topk)
                if snap.recv != want:
                    print(
                        f"[rank {rank}] {tag} {api}: recv={snap.recv}, want {want}",
                        flush=True,
                    )
                    failures += 1
                failures += _check_against_source(d, snap, gathered, m, f"{tag} {api}")
                if not capped:
                    combined[api] = _combine(op, snap, table, dev)[:n].cpu()
                lockstep()

            # Same rows from the same origins, whatever the slot order.
            f_orig, f_rows = snaps["fused"].by_origin()
            s_orig, s_rows = snaps["split"].by_origin()
            if not capped:
                same = torch.equal(f_orig, s_orig) and all(
                    torch.equal(f_rows[k], s_rows[k]) for k in f_rows
                )
                if not same:
                    print(f"[rank {rank}] {tag}: split rows != fused rows", flush=True)
                    failures += 1
                if not torch.equal(snaps["fused"].dest_pe, snaps["split"].dest_pe):
                    print(f"[rank {rank}] {tag}: split tok_map != fused", flush=True)
                    failures += 1
                if not torch.equal(combined["fused"], combined["split"]):
                    print(f"[rank {rank}] {tag}: split combine != fused", flush=True)
                    failures += 1
                if n:
                    uniq = _distinct_pes(gathered[rank][1], epr)
                    exp = (table[rank * m : rank * m + n].float().cpu() * uniq).to(
                        torch.bfloat16
                    )
                    got = combined["split"]
                    ok = (
                        torch.allclose(got.float(), exp.float(), atol=0.5, rtol=0.1)
                        if quantized
                        else torch.equal(got, exp)
                    )
                    if not ok:
                        print(f"[rank {rank}] {tag}: combine round trip", flush=True)
                        failures += 1

            # Replay the split round's routing with a new payload: the slots are
            # fixed now, so split and fused must agree position by position.
            x2 = _payload(disp, n, hidden, gen, dev)
            w2 = torch.rand(n, topk, generator=gen, device=dev)
            s2 = _scales(scale, n, hidden, gen, dev)
            handle = snaps["split"].routing
            rep = {}
            views = (op.recv_tokens(), op.recv_weights(), op.recv_indices())
            views += () if scale is None else (op.recv_scales(),)
            for api in ("fused", "split"):
                # Both replays start from a zeroed arena, so rows past the count
                # agree and a split that wrote nothing cannot pass.
                for t in views:
                    t.zero_()
                lockstep()
                if api == "fused":
                    ret = op.dispatch(x2, w2, s2, idx, routing=handle)
                else:
                    ret = op.dispatch_send(x2, w2, s2, idx, routing=handle)
                    op.dispatch_recv()
                lockstep()
                out, rw, rs, ri, total = ret
                rep[api] = [int(total[0].item()), _bytes(out[:cap]).cpu()]
                rep[api] += [rw[:cap].cpu(), ri[:cap].cpu()]
                if rs is not None:
                    rep[api].append(rs[:cap].cpu())
            if rep["fused"][0] != want or not all(
                (a == b) if isinstance(a, int) else torch.equal(a, b)
                for a, b in zip(rep["fused"], rep["split"])
            ):
                print(f"[rank {rank}] {tag}: split replay != fused replay", flush=True)
                failures += 1
    return failures


def run_stress(d, comm, rounds):
    """Rounds with no host sync; split/fused mixed; work between send and recv."""
    hidden, topk, epr, m = 4096, 6, 16, 64
    rank, world, dev = d.rank, d.world, d.dev
    cfg = _config(d, hidden, topk, epr, m, "bf16", None, {})
    plans = []
    for p in range(8):
        n = ((rank + p) * 5) % (m + 1)
        gen = torch.Generator(device=dev).manual_seed(4321 + rank * 31 + p)
        x = torch.randn(n, hidden, generator=gen, device=dev).to(torch.bfloat16)
        w = torch.rand(n, topk, generator=gen, device=dev)
        idx = _routing(world, m, topk, epr, gen, dev)[:n]
        gathered = _gather(n, idx, x, w, None, world)
        want = torch.tensor([_expected_count(rank, gathered, epr)], device=dev)
        uniq = _distinct_pes(gathered[rank][1], epr).to(dev)
        exp = (x.float() * uniq).to(torch.bfloat16)
        plans.append((x, w, idx, want.to(torch.int32), exp))
    a = torch.randn(1024, 1024, device=dev).to(torch.bfloat16)
    lockstep()

    failures = 0
    with EpDispatchCombineOp(cfg, comm) as op:
        bad = torch.zeros(1, dtype=torch.int32, device=dev)
        for r in range(rounds):
            x, w, idx, want, exp = plans[r % len(plans)]
            if r < rounds // 2:
                split = r % 2 == 0
            else:  # each rank on its own: the two APIs share one protocol
                split = (((r * 2654435761) ^ (rank * 40503)) & 4) != 0
            if split:
                out, _, _, _, total, routing = op.dispatch_send(
                    x, w, None, idx, return_routing=True
                )
                # A different rank is slow each round, so peers wait in recv.
                lag = (r * 7 + rank * 3) % 11
                if lag == 0:
                    torch.cuda._sleep(2_000_000)
                elif lag < 4:
                    for _ in range(lag):
                        a = (a @ a).clamp_(-1, 1)
                op.dispatch_recv()
            else:
                out, _, _, _, total, routing = op.dispatch(
                    x, w, None, idx, return_routing=True
                )
            bad += (total != want).to(torch.int32)
            cout, _ = op.combine(out, routing=routing)
            n = x.shape[0]
            if n:
                bad += (cout[:n] != exp).any().to(torch.int32)
        lockstep()
        if int(bad.item()):
            print(f"[rank {rank}] stress: {int(bad.item())} bad rounds", flush=True)
            failures += 1
        recv_num = from_gpu_ptr(
            op.arena.local_ptr("recv_num"), (world,), torch.int32
        ).cpu()
        tok_off = from_gpu_ptr(op.arena.local_ptr("tok_off"), (1,), torch.int32).cpu()
        ctr = op.dest_pe_counter.cpu()
        if recv_num.any() or tok_off.any() or ctr.any():
            print(
                f"[rank {rank}] stress: counters left set: recv_num={recv_num.tolist()} "
                f"tok_off={tok_off.tolist()} dest_pe_counter={ctr.tolist()}",
                flush=True,
            )
            failures += 1
    return failures


def run_call_order(d, comm):
    """Out-of-order calls are refused before launching; the op stays usable."""
    hidden, topk, epr, m = 2048, 8, 8, 32
    rank, world, dev = d.rank, d.world, d.dev
    cfg = _config(d, hidden, topk, epr, m, "bf16", None, {})
    gen = torch.Generator(device=dev).manual_seed(777 + rank)
    x = torch.randn(m, hidden, generator=gen, device=dev).to(torch.bfloat16)
    w = torch.rand(m, topk, generator=gen, device=dev)
    idx = _routing(world, m, topk, epr, gen, dev)
    gathered = _gather(m, idx, x, w, None, world)
    want = _expected_count(rank, gathered, epr)
    failures = 0

    def refused(tag, fn, kind=RuntimeError):
        nonlocal failures
        try:
            fn()
        except kind:
            return
        except Exception as e:  # the wrong refusal is a failure too
            print(f"[rank {rank}] order {tag}: {type(e).__name__}: {e}", flush=True)
        else:
            print(f"[rank {rank}] order {tag}: was accepted", flush=True)
        failures += 1

    def verified_round(tag, split):
        nonlocal failures
        if split:
            ret = op.dispatch_send(x, w, None, idx, return_routing=True)
            op.dispatch_recv()
        else:
            ret = op.dispatch(x, w, None, idx, return_routing=True)
        lockstep()
        snap = _Snapshot(op, ret, m, topk)
        if snap.recv != want:
            print(
                f"[rank {rank}] order {tag}: recv={snap.recv} want {want}", flush=True
            )
            failures += 1
        failures += _check_against_source(d, snap, gathered, m, f"order {tag}")
        op.combine(ret[0][: snap.recv].clone(), routing=snap.routing)
        lockstep()

    with EpDispatchCombineOp(cfg, comm) as op:
        refused("dispatch_recv while idle", op.dispatch_recv)
        refused("combine_send", lambda: op.combine_send(x, routing=None), Exception)
        refused("combine_recv", op.combine_recv, Exception)
        over = torch.zeros(m + 1, hidden, dtype=torch.bfloat16, device=dev)
        refused(
            "send over capacity",
            lambda: op.dispatch_send(over, w, None, idx),
            ValueError,
        )
        verified_round("first round after refusals", split=True)

        *_, routing = op.dispatch_send(x, w, None, idx, return_routing=True)
        refused("dispatch_send twice", lambda: op.dispatch_send(x, w, None, idx))
        refused("dispatch while sent", lambda: op.dispatch(x, w, None, idx))
        refused("combine while sent", lambda: op.combine(x, routing=routing))
        refused("local_expert_count while sent", op.local_expert_count)
        op.dispatch_recv()
        lockstep()
        refused("dispatch_recv twice", op.dispatch_recv)
        op.local_expert_count()
        op.combine(op.recv_tokens(), routing=routing)
        lockstep()
        verified_round("after a refused round", split=False)

        # Leave a round open and recover with reset(). Every rank's send must be
        # done first (lockstep), or a late peer's count lands after the zeroing.
        op.dispatch_send(x, w, None, idx)
        refused("dispatch with a split open", lambda: op.dispatch(x, w, None, idx))
        lockstep()
        op.reset()
        lockstep()
        verified_round("after reset", split=True)
        verified_round("second round after reset", split=False)
    return failures


def run_graph(d, comm, empty_all, replays):
    """send -> GEMM -> recv -> combine, captured once, replayed with new data."""
    hidden, topk, epr, m = 2048, 6, 8, 16
    rank, world, dev = d.rank, d.world, d.dev
    n = 0 if empty_all or rank == 0 else m
    cfg = _config(d, hidden, topk, epr, m, "bf16", None, {})
    idx_cpu = torch.arange(n * topk, dtype=torch.int32).view(n, topk) % (world * epr)
    indices = idx_cpu.to(dev)
    base_x = (
        (torch.arange(n).view(n, 1) % 2)
        + (torch.arange(hidden).view(1, hidden) % 2)
        + 2 * (rank % 4)
        + 1
    )
    base_w = torch.arange(n * topk, dtype=torch.float32).view(n, topk) + rank * 1e5 + 1
    x = torch.empty((n, hidden), dtype=torch.bfloat16, device=dev)
    weights = torch.empty((n, topk), dtype=torch.float32, device=dev)
    uniq = _distinct_pes(idx_cpu, epr)
    a = torch.randn(512, 512, device=dev).to(torch.bfloat16)
    failures = 0
    scenario = "all_zero" if empty_all else "zero_source_receives"
    with EpDispatchCombineOp(cfg, comm) as op:
        last = {}

        def update(generation):
            x.copy_((base_x + generation % 32).to(torch.bfloat16))
            weights.copy_(base_w + generation * 1024)

        def step():
            ret = op.dispatch_send(x, weights, None, indices, return_routing=True)
            last["gemm"] = a @ a  # the shared experts' slot
            op.dispatch_recv()
            # The lazy reverse-map clone is captured after recv, so each replay
            # copies that replay's map.
            reverse = ret[5].disp_tok_id_to_src_tok_id_local
            cout, _ = op.combine(ret[0], routing=ret[5])
            last.update(ret=ret, reverse=reverse, combined=cout)

        def check(generation):
            lockstep()
            gathered = _gather(n, indices, x, weights, None, world)
            out, rw, _, ri, total, routing = last["ret"]
            recv = int(total[0].item())
            tag = f"graph={scenario} generation={generation}"
            bad = 0
            want = _expected_count(rank, gathered, epr)
            if recv != want:
                print(f"[rank {rank}] {tag}: recv={recv} want {want}", flush=True)
                bad += 1
            snap = _Snapshot.__new__(_Snapshot)
            snap.recv = recv
            snap.reverse = last["reverse"][:recv].cpu()
            snap.rows = {
                "tokens": _bytes(out[:recv]).cpu(),
                "weights": rw[:recv].cpu(),
                "indices": ri[:recv].cpu(),
            }
            bad += _check_against_source(d, snap, gathered, m, tag)
            exp = (x.float().cpu() * uniq).to(torch.bfloat16)
            if not torch.equal(last["combined"][:n].cpu(), exp):
                print(f"[rank {rank}] {tag}: identity round trip failed", flush=True)
                bad += 1
            return bad

        update(0)
        step()
        failures += check("prime")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        lockstep()
        for generation in range(1, replays + 1):
            update(generation)
            lockstep()
            graph.replay()
            failures += check(generation)
        del graph
        lockstep()
    return failures


def run_perf(d, comm, iters):
    """Per token count: fused vs split, and the overlap with a GEMM in between."""
    hidden, topk, epr, m = 7168, 6, 48, 512  # DeepSeek-V4: 384 experts, top-6
    rank, world, dev = d.rank, d.world, d.dev
    disp = "fp4" if _arch().startswith("gfx95") else "bf16"
    scale = (32, 1) if disp == "fp4" else None
    cfg = _config(d, hidden, topk, epr, m, disp, scale, {})
    # A shared expert's two GEMMs (gate_up, then down), DeepSeek-V4-sized.
    w1 = torch.randn(hidden, 2 * 2048, device=dev).to(torch.bfloat16)
    w2 = torch.randn(2048, hidden, device=dev).to(torch.bfloat16)
    with EpDispatchCombineOp(cfg, comm) as op:
        # combine() skips its staging copy for its own view, so only the
        # transport is timed.
        stage = op.combine_in_view()
        for n in (16, 64, 128, 512):
            gen = torch.Generator(device=dev).manual_seed(5 + rank)
            x = _payload(disp, n, hidden, gen, dev)
            s = _scales(scale, n, hidden, gen, dev)
            w = torch.rand(n, topk, generator=gen, device=dev)
            idx = _routing(world, m, topk, epr, gen, dev)[:n]
            h = torch.randn(n, hidden, device=dev).to(torch.bfloat16)

            def shared():
                return (h @ w1)[:, :2048] @ w2

            def fused(gemm):
                *_, routing = op.dispatch(x, w, s, idx, return_routing=True)
                if gemm:
                    shared()
                op.combine(stage, routing=routing)

            def split(gemm):
                *_, routing = op.dispatch_send(x, w, s, idx, return_routing=True)
                if gemm:
                    shared()
                op.dispatch_recv()
                op.combine(stage, routing=routing)

            times = {}
            for label, fn, gemm in (
                ("fused", fused, False),
                ("split", split, False),
                ("fused+gemm", fused, True),
                ("send+gemm+recv", split, True),
            ):
                for _ in range(10):
                    fn(gemm)
                lockstep()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    fn(gemm)
                end.record()
                lockstep()
                times[label] = d.allreduce_max(start.elapsed_time(end) * 1e3 / iters)
            if rank == 0:
                gain = times["fused+gemm"] - times["send+gemm+recv"]
                print(
                    f"# perf {disp} n={n}: fused {times['fused']:.1f} us, split "
                    f"{times['split']:.1f} us; fused+gemm {times['fused+gemm']:.1f} "
                    f"us, send+gemm+recv {times['send+gemm+recv']:.1f} us "
                    f"(saves {gain:.1f} us/layer)",
                    flush=True,
                )


def main():
    d = Dist()
    modes = os.environ.get("MODES")
    modes = modes.split(",") if modes else _default_modes()
    loops = max(3, int(os.environ.get("LOOP", "3")))
    stress = int(os.environ.get("STRESS_ROUNDS", "1000"))
    replays = int(os.environ.get("GRAPH_REPLAYS", "32"))
    perf = int(os.environ.get("PERF_ITERS", "100"))

    # One communicator per scenario: several differently-sized windows in one
    # reservation trip the ROCm 7.0-7.2 clr bug test_asyncll.py describes.
    def own_comm(run, hidden, m):
        uid = Communicator.get_unique_id() if d.rank == 0 else None
        uid = d.bcast_uid(uid)
        win = d.world * hidden * m * 2 * 4 + (1 << 24)
        with Communicator.init(
            d.world, d.rank, uid, per_rank_vmm=6 * win + (1 << 28)
        ) as comm:
            return run(comm)

    failures = 0
    for name in modes:
        hidden, _, _, m, *_ = _MODES[name]
        failures += own_comm(lambda c: run_mode(d, c, name, loops), hidden, m)
        if d.rank == 0:
            print(f"# native split {name}: {loops} iterations done", flush=True)
    if stress:
        failures += own_comm(lambda c: run_stress(d, c, stress), 4096, 64)
        if d.rank == 0:
            print(f"# native split stress: {stress} rounds done", flush=True)
    failures += own_comm(lambda c: run_call_order(d, c), 2048, 32)
    if d.rank == 0:
        print("# native split call order: done", flush=True)
    if replays:
        for empty_all in (False, True):
            failures += own_comm(
                lambda c: run_graph(d, c, empty_all, replays), 2048, 16
            )
            if d.rank == 0:
                print(
                    f"# native split graph empty_all={empty_all}: {replays} replays",
                    flush=True,
                )
    if perf:
        own_comm(lambda c: run_perf(d, c, perf), 7168, 512)

    total = d.allreduce_sum(failures)
    if d.rank == 0:
        print(
            f"# {'PASS' if total == 0 else 'FAIL'}: {total} failed checks across "
            f"{d.world} ranks, modes {','.join(modes)} x {loops} iterations, "
            f"{stress} stress rounds, call order, 2 graph scenarios x {replays} "
            "replays",
            flush=True,
        )
    dist.destroy_process_group()
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
