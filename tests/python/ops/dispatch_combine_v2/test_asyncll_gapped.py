# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""AsyncLL gapped layout (dispatch_send(gap_stride=...)) correctness on one node.

    MORI_ENABLE_SDMA=1 MORI_ASYNCLL_EARLY_NOTIFY=1 \\
        torchrun --standalone --nproc_per_node=8 test_asyncll_gapped.py

Env: ROUNDS (default 8), SHAPES "hidden,topk,epr,m;..." to override.

Per gapped round, on rank r:
  * segment s (rows [s*S, s*S + c_s)) holds exactly the tokens rank s routed to r,
    each once, with hidden / indices / weights / scales equal to the source row;
  * the rows [s*S + c_s, (s+1)*S) route only to experts r does not own, weight 0;
  * an identity expert written into combine_in_view() round-trips: each token
    comes back scaled by its number of distinct destination ranks.
Gapped and packed rounds alternate on one op, so each layout must survive the
other having used the same arena. Weights column 0 carries (rank, token) so a
received row identifies its source.
"""

import os
import sys

import torch
import torch.distributed as dist
from mori.cco.communicator import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

# (hidden, topk, experts_per_rank, max_tokens, fp8 dispatch with e8m0-like scales)
_DEFAULT_SHAPES = [
    (2048, 8, 8, 64, False),
    (7168, 6, 48, 256, True),  # DeepSeek-V4 MoE shape, mxfp8 dispatch
    (4096, 6, 16, 96, True),
]


def _shapes():
    raw = os.environ.get("SHAPES")
    if not raw:
        return _DEFAULT_SHAPES
    rows = []
    for row in raw.split(";"):
        h, k, e, m, f = (int(x) for x in row.split(","))
        rows.append((h, k, e, m, bool(f)))
    return rows


def _encode(rank, token):
    return float(rank * 65536 + token)


def _gather(obj, world):
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out


def _round(op, rank, world, hidden, topk, epr, m, fp8, it, dev):
    gen = torch.Generator(device=dev).manual_seed(4321 + rank * 31 + it)
    cpu_gen = torch.Generator().manual_seed(99 + it)
    counts = torch.randint(0, m + 1, (world,), generator=cpu_gen).tolist()
    if it % 4 == 3:
        counts[(it // 4) % world] = 0
    gapped = it % 3 != 1
    slack = 7 if it % 2 else 0
    stride = min(m, max(max(counts), 1) + slack)
    n = counts[rank]

    x = torch.randn(n, hidden, generator=gen, device=dev, dtype=torch.bfloat16)
    if fp8:
        x = x.to(torch.float8_e4m3fn)
        scales = torch.randint(
            0, 255, (n, hidden // 32), generator=gen, device=dev, dtype=torch.uint8
        )
    else:
        scales = None
    w = torch.rand(n, topk, generator=gen, device=dev, dtype=torch.float32)
    w[:, 0] = torch.tensor([_encode(rank, t) for t in range(n)], device=dev)
    idx = torch.stack(
        [
            torch.randperm(epr * world, generator=gen, device=dev)[:topk]
            for _ in range(n)
        ]
    ).to(torch.int32) if n else torch.empty(0, topk, dtype=torch.int32, device=dev)

    gathered = _gather(
        (
            idx.cpu(),
            x.view(torch.uint8).cpu() if fp8 else x.cpu(),
            w.cpu(),
            None if scales is None else scales.cpu(),
        ),
        world,
    )
    kwargs = {"gap_stride": stride} if gapped else {}
    out, out_w, out_s, out_idx, total, routing = op.dispatch_send(
        x, w, scales, idx, return_routing=True, **kwargs
    )
    op.dispatch_recv()
    torch.cuda.synchronize()
    dist.barrier()

    tag = f"h={hidden} k={topk} fp8={fp8} it={it} gapped={gapped} S={stride}"
    failures = 0
    if gapped:
        failures += _check_gapped(
            rank, world, epr, stride, out, out_w, out_s, out_idx, total, gathered, fp8, tag
        )
        rows = world * stride
    else:
        rows = int(total[0].item())

    cin = op.combine_in_view()
    if gapped and cin.shape[0] != world * stride:
        print(f"[rank {rank}] {tag}: combine_in_view rows {cin.shape[0]}", flush=True)
        failures += 1
    # Identity expert: the dispatched row itself, in the combine dtype.
    cin[:rows].copy_(out[:rows].to(torch.bfloat16))
    cout, _ = op.combine_send(cin[:rows], None, idx, routing=routing)
    op.combine_recv()
    torch.cuda.synchronize()
    dist.barrier()
    if n:
        uniq = torch.tensor(
            [len({int(e) // epr for e in row.tolist()}) for row in gathered[rank][0]],
            dtype=torch.float32,
        )
        src = x.to(torch.bfloat16).float().cpu()
        want = (src * uniq[:, None]).to(torch.bfloat16).float()
        got = cout[:n].float().cpu()
        if not torch.allclose(got, want, atol=1e-2, rtol=1e-2):
            diff = (got - want).abs().max().item()
            print(f"[rank {rank}] {tag}: combine max |diff| {diff:.4f}", flush=True)
            failures += 1
    return failures


def _check_gapped(rank, world, epr, stride, out, out_w, out_s, out_idx, total, gathered, fp8, tag):
    failures = 0
    rows = world * stride
    if out.shape[0] != rows or int(total[0].item()) != rows:
        print(f"[rank {rank}] {tag}: rows {out.shape[0]} total {int(total[0])}", flush=True)
        return 1
    got_x = (out.view(torch.uint8) if fp8 else out).cpu()
    got_w, got_idx = out_w.cpu(), out_idx.cpu()
    got_s = None if out_s is None else out_s.contiguous().view(torch.uint8).cpu()
    for s, (s_idx, s_x, s_w, s_scales) in enumerate(gathered):
        mine = [
            t for t in range(s_idx.shape[0])
            if rank in {int(e) // epr for e in s_idx[t].tolist()}
        ]
        lo, c = s * stride, len(mine)
        seg_w = got_w[lo : lo + c]
        seen = sorted(int(v) - s * 65536 for v in seg_w[:, 0].tolist())
        if seen != mine:
            print(f"[rank {rank}] {tag}: segment {s} tokens {seen[:8]} want {mine[:8]}", flush=True)
            failures += 1
            continue
        order = seg_w[:, 0].long() - s * 65536
        fields = {
            "hidden": (got_x[lo : lo + c], s_x[order]),
            "indices": (got_idx[lo : lo + c], s_idx[order]),
            "weights": (seg_w, s_w[order]),
        }
        if s_scales is not None:
            fields["scales"] = (got_s[lo : lo + c, : s_scales.shape[1]], s_scales[order])
        for name, (got, want) in fields.items():
            if not torch.equal(got, want):
                print(f"[rank {rank}] {tag}: segment {s} {name} mismatch", flush=True)
                failures += 1
        gap_idx = got_idx[lo + c : lo + stride]
        gap_w = got_w[lo + c : lo + stride]
        if gap_idx.numel() and (
            (gap_idx // epr == rank).any().item() or (gap_w != 0).any().item()
        ):
            print(f"[rank {rank}] {tag}: segment {s} gap rows reach a local expert", flush=True)
            failures += 1
    return failures


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    rounds = int(os.environ.get("ROUNDS", "8"))
    failures = 0
    for hidden, topk, epr, m, fp8 in _shapes():
        objs = [Communicator.get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        win_bytes = world * hidden * m * 2 * 4 + (1 << 24)
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=world,
            hidden_dim=hidden,
            max_num_inp_token_per_rank=m,
            num_experts_per_rank=epr,
            num_experts_per_token=topk,
            data_type=torch.bfloat16,
            dispatch_data_type=torch.float8_e4m3fn if fp8 else None,
            combine_data_type=torch.bfloat16 if fp8 else None,
            scale_dim=hidden // 32 if fp8 else 0,
            scale_type_size=1 if fp8 else 0,
            kernel_backend="hip",
            async_ll=True,
            async_ll_transport="sdma",
        )
        with Communicator.init(
            world, rank, objs[0], per_rank_vmm=6 * win_bytes + (1 << 28)
        ) as comm, EpDispatchCombineOp(cfg, comm) as op:
            for it in range(rounds):
                failures += _round(op, rank, world, hidden, topk, epr, m, fp8, it, dev)
        if rank == 0:
            print(f"# gapped hidden={hidden} topk={topk} epr={epr} m={m} fp8={fp8}: done", flush=True)
    t = torch.tensor([failures], dtype=torch.int64)
    dist.all_reduce(t)
    if rank == 0:
        print(f"# {'PASS' if t.item() == 0 else 'FAIL'}: {t.item()} failed checks", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if t.item() == 0 else 1)


if __name__ == "__main__":
    main()
