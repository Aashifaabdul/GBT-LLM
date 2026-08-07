"""
Joint, differentiable training of GBTICLNet (the real GBT-ICL model) and
TinyTransformerCoeffModel (the real LLM-style coefficient predictor),
end to end, against real extracted video frames.

REQUIRES PyTorch. This script was written and syntax-checked in an
environment without PyTorch installed (see the project README) and has NOT
been executed end to end -- run it on your GPU machine, and treat the first
run as a smoke test on a small --samples-per-frame / --epochs before a full
training run. Everything downstream (run_dataset_pipeline.py) depends on the
checkpoint this script produces being real and correct, so please verify the
loss is actually decreasing before trusting a checkpoint.

============================== What is trained =============================
Per sampled 8x8 block (context = its true top row + left column from the
real image -- exactly what a correctly-functioning decoder would have
already reconstructed at that point):

  1. GBTICLNet predicts edge weights from context -> Laplacian -> eigh
     -> (eigvals, U).  [replaces the hand-written ContextGradientGBTICL formula]
  2. forward_gft_batch(real_block, U) -> real coefficients.
  3. Straight-through quantization: q_hard = round(coeffs / step), with a
     straight-through estimator so gradients still flow through the rounding
     step back into GBTICLNet (standard trick in learned compression --
     rounding itself has zero gradient almost everywhere, so we substitute
     the identity gradient locally).
  4. TinyTransformerCoeffModel.forward_sequence(eigvals, q_hard) -> logits
     predicting each quantized coefficient from Λ + coefficient history,
     teacher-forced (parallel, like training any causal LM).
     [replaces the hand-written LaplaceCoeffModel formula]

============================== The loss ======================================
  distortion = MSE(inverse_gft(dequantize(q_soft)), real_block)   -- via STE
  rate       = mean cross-entropy(logits, true_symbol_index) in bits
               (this *is* the exact expected coding cost the real range
               coder will pay for a categorical model this accurate -- no
               approximation needed for the rate term, only for distortion's
               quantization step)
  loss = distortion + LAMBDA_RATE * rate

LAMBDA_RATE trades compression ratio against quality, same role as `lambda`
in any rate-distortion learned codec -- higher = smaller files, lower PSNR.
Train a few checkpoints at different LAMBDA_RATE values if you want a proper
rate-distortion curve for your dissertation results (this is standard
practice and expected in the write-up, not extra work you're inventing).

======================= Known numerical caveat (read this) ===================
GBTICLNet can predict a near-uniform graph, which produces a Laplacian with
repeated/degenerate eigenvalues (this was directly observed and debugged for
UniformGBTICL earlier: ~31 of 64 eigenvalues repeated). torch.linalg.eigh's
backward pass is well-behaved for eigenvalue gradients even then, but
eigenvector (U) gradients can spike in that regime. This script clips
gradient norms (see --grad-clip) specifically to keep this from derailing
training; if you see loss spikes or NaNs, lower --lr and/or --grad-clip
further before assuming there's a bug elsewhere.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise SystemExit(
        "training.py needs PyTorch. This was written/syntax-checked in an "
        "environment without it installed -- run this on your GPU machine. "
        f"(import error: {e})"
    )

from PIL import Image

from gbticl_pipeline.device_utils import get_device
from gbticl_pipeline.graph_model import GBTICLNet, edge_list
from gbticl_pipeline.graph_utils import build_laplacian_batch, eigendecompose
from gbticl_pipeline.gft import forward_gft_batch, inverse_gft_batch
from gbticl_pipeline.coeff_model import TinyTransformerCoeffModel

BLOCK_SIZE = 8
PAD_VALUE = 128


# --------------------------------------------------------------------------
# Data: sample real (context, block) pairs from the already-extracted frames.
# Context here is the TRUE neighbouring pixels from the real image -- valid
# because at inference the decoder's already-reconstructed neighbours are,
# by construction of a working codec, a close approximation of the true
# pixels (exactly true at quant_step -> 0, and the whole point of training
# against real data is to make this hold up well away from that limit too).
# --------------------------------------------------------------------------

class BlockContextDataset(Dataset):
    def __init__(self, frame_paths, block_size=BLOCK_SIZE, samples_per_frame=3000, seed=0):
        self.block_size = block_size
        self.items = []  # list of (top, left, valid_top, valid_left, block) numpy arrays
        rng = np.random.default_rng(seed)

        for p in frame_paths:
            img = np.array(Image.open(p).convert("RGB"))
            h, w = img.shape[:2]
            n_bh, n_bw = h // block_size, w // block_size
            all_idx = [(i, j) for i in range(n_bh) for j in range(n_bw)]
            n_take = min(samples_per_frame, len(all_idx))
            chosen = rng.choice(len(all_idx), size=n_take, replace=False)

            for c in chosen:
                i, j = all_idx[c]
                r0, c0 = i * block_size, j * block_size
                block = img[r0:r0 + block_size, c0:c0 + block_size, :].copy()

                if i > 0:
                    top = img[r0 - 1, c0:c0 + block_size, :].copy()
                    valid_top = True
                else:
                    top = np.full((block_size, 3), PAD_VALUE, dtype=np.uint8)
                    valid_top = False

                if j > 0:
                    left = img[r0:r0 + block_size, c0 - 1, :].copy()
                    valid_left = True
                else:
                    left = np.full((block_size, 3), PAD_VALUE, dtype=np.uint8)
                    valid_left = False

                self.items.append((top, left, valid_top, valid_left, block))

        print(f"BlockContextDataset: {len(self.items)} sampled blocks from {len(frame_paths)} frames")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        top, left, vt, vl, block = self.items[idx]
        return (
            torch.from_numpy(top), torch.from_numpy(left),
            torch.tensor(vt), torch.tensor(vl),
            torch.from_numpy(block),
        )


# --------------------------------------------------------------------------
# Training step
# --------------------------------------------------------------------------

def train_step(gbticl_net, coeff_net, batch, quant_step, symbol_range, lambda_rate, device):
    top, left, valid_top, valid_left, block = batch
    top, left, block = top.to(device), left.to(device), block.to(device)
    valid_top, valid_left = valid_top.to(device), valid_left.to(device)

    feats = gbticl_net.features_batch(top, left, valid_top, valid_left, device)
    logits_w = gbticl_net.forward(feats)
    weights = gbticl_net.to_weights(logits_w)            # (B, n_edges)

    L = build_laplacian_batch(weights, gbticl_net.block_size)   # (B, n, n)
    eigvals, U = eigendecompose(L)                               # (B,n), (B,n,n)

    coeffs = forward_gft_batch(block, U)                  # (B, n, 3) float64

    # --- straight-through quantization ---
    coeffs_scaled = coeffs / quant_step
    q_hard = torch.round(coeffs_scaled)
    q_ste = coeffs_scaled + (q_hard - coeffs_scaled).detach()   # forward=hard, backward=identity

    recon = inverse_gft_batch(q_ste * quant_step, U, gbticl_net.block_size)  # (B, bs, bs, 3)
    distortion = F.mse_loss(recon, block.to(torch.float64))

    # --- rate: exact expected bits under the categorical model ---
    lo, hi = symbol_range
    n_symbols = hi - lo + 1
    b = q_hard.shape[0]

    q_int = torch.clamp(q_hard, lo, hi).to(torch.int64)     # (B, n, 3)
    target_idx = (q_int - lo)                                # (B, n, 3), class indices

    # (B, n) -> (B, 3, n) -> (B*3, n): each block's eigenvalue spectrum is shared
    # across its 3 channels (same graph/eigendecomposition, per codec.py's design),
    # but every channel gets its own coefficient history -- so we flatten
    # (block, channel) into one training-batch axis, channel-major within each
    # block, and lay q_int/target_idx out (B, n, 3) -> (B, 3, n) the same way so
    # row r of every _rep tensor refers to the same (block, channel).
    eigvals_rep = eigvals.unsqueeze(1).expand(-1, 3, -1).reshape(b * 3, -1)      # (B*3, n)
    true_vals_rep = q_int.permute(0, 2, 1).reshape(b * 3, -1).to(torch.float32)  # (B*3, n)
    target_rep = target_idx.permute(0, 2, 1).reshape(b * 3, -1)                  # (B*3, n)

    coeff_logits = coeff_net.forward_sequence(eigvals_rep, true_vals_rep)  # (B*3, n, n_symbols)
    ce_nats = F.cross_entropy(
        coeff_logits.reshape(-1, n_symbols), target_rep.reshape(-1), reduction="mean"
    )
    rate_bits = ce_nats / np.log(2.0)  # nats -> bits

    loss = distortion + lambda_rate * rate_bits
    return loss, distortion.item(), rate_bits.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dirs", nargs="+", default=["Beauty/frames", "HoneyBee/frames"],
                     help="One or more directories of extracted PNG frames.")
    ap.add_argument("--samples-per-frame", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                     help="Gradient-norm clip -- see the eigh-degeneracy caveat in the module docstring.")
    ap.add_argument("--quant-step", type=float, default=8.0)
    ap.add_argument("--symbol-lo", type=int, default=-2200)
    ap.add_argument("--symbol-hi", type=int, default=2200)
    ap.add_argument("--lambda-rate", type=float, default=0.01,
                     help="Rate-distortion trade-off. Train several checkpoints at different "
                          "values to trace out a rate-distortion curve for your results section.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/gbticl_ckpt.pt")
    ap.add_argument("--num-workers", type=int, default=2,
                     help="DataLoader worker processes. 0 = load in the main process (slower).")
    ap.add_argument("--log-every", type=int, default=20,
                     help="Print running loss every N batches -- without this, nothing prints "
                          "until a full epoch (thousands of batches) finishes, which looks like "
                          "a hang even when training is progressing normally.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"training on device: {device}")

    script_dir = Path(__file__).resolve().parent
    frame_paths = []
    for d in args.frames_dirs:
        d_path = (script_dir / d) if not Path(d).is_absolute() else Path(d)
        frame_paths += sorted(d_path.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"No PNG frames found under {args.frames_dirs} (resolved against {script_dir})")
    print(f"found {len(frame_paths)} frames")

    dataset = BlockContextDataset(frame_paths, BLOCK_SIZE, args.samples_per_frame, args.seed)
    steps_per_epoch = len(dataset) // args.batch_size
    print(f"{steps_per_epoch} optimizer steps/epoch x {args.epochs} epochs = "
          f"{steps_per_epoch*args.epochs} total steps. Each step does a batched "
          f"{args.batch_size}x(64x64) eigendecomposition (forward+backward) -- this is the "
          f"expensive part; --log-every {args.log_every} batches will confirm it's progressing.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    symbol_range = (args.symbol_lo, args.symbol_hi)
    gbticl_net = GBTICLNet(block_size=BLOCK_SIZE).to(device)
    coeff_net = TinyTransformerCoeffModel(block_size=BLOCK_SIZE, symbol_range=symbol_range).to(device)

    params = list(gbticl_net.parameters()) + list(coeff_net.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"total trainable parameters: {n_params:,} "
          f"(GBTICLNet: {sum(p.numel() for p in gbticl_net.parameters()):,}, "
          f"TinyTransformerCoeffModel: {sum(p.numel() for p in coeff_net.parameters()):,})")

    optimizer = torch.optim.Adam(params, lr=args.lr)

    out_path = script_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        ep_loss = ep_dist = ep_rate = 0.0
        run_loss = run_dist = run_rate = 0.0
        n_batches = 0
        for batch in loader:
            step_t0 = time.time()
            optimizer.zero_grad()
            loss, dist, rate = train_step(
                gbticl_net, coeff_net, batch, args.quant_step, symbol_range, args.lambda_rate, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            ep_loss += loss.item(); ep_dist += dist; ep_rate += rate
            run_loss += loss.item(); run_dist += dist; run_rate += rate
            n_batches += 1

            if n_batches % args.log_every == 0:
                step_s = (time.time() - step_t0)
                eta_min = step_s * (steps_per_epoch - n_batches) / 60
                print(f"  epoch {epoch+1} batch {n_batches}/{steps_per_epoch}  "
                      f"loss={run_loss/args.log_every:.4f}  "
                      f"distortion={run_dist/args.log_every:.4f}  "
                      f"rate={run_rate/args.log_every:.4f}  "
                      f"~{step_s:.2f}s/batch  ETA this epoch ~{eta_min:.1f}min")
                run_loss = run_dist = run_rate = 0.0

        ep_loss /= n_batches; ep_dist /= n_batches; ep_rate /= n_batches
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs}  loss={ep_loss:.4f}  "
              f"distortion(mse)={ep_dist:.4f}  rate(bits/coeff)={ep_rate:.4f}  "
              f"[{dt:.1f}s, {n_batches} batches]")
        history.append(dict(epoch=epoch + 1, loss=ep_loss, distortion=ep_dist, rate=ep_rate, seconds=dt))

        torch.save({
            "gbticl_net": gbticl_net.state_dict(),
            "coeff_net": coeff_net.state_dict(),
            "block_size": BLOCK_SIZE,
            "symbol_range": symbol_range,
            "quant_step": args.quant_step,
            "lambda_rate": args.lambda_rate,
            "epoch": epoch + 1,
            "history": history,
        }, out_path)
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
