"""
Rate-distortion training of GBT-ICL and the coefficient model on extracted frames.

For every sampled 8x8 block the graph model predicts edge weights, from which the
Laplacian eigenbasis U and spectrum are computed. The block is transformed with U
and quantised with a straight-through estimator (rounding in the forward pass,
identity gradient in the backward pass). The loss is

    loss = MSE(reconstruction, block) + lambda_rate * rate

where rate is the cross-entropy in bits per coefficient of the coefficient model
(teacher-forced), i.e. the expected range-coder cost. Gradient norms are clipped
(--grad-clip) because eigenvector gradients can spike when the predicted graph is
close to uniform (repeated eigenvalues).

Staged training (each stage is a separate run):

  Stage A: meta-train GBTICLMetaLearner against the closed-form Laplace rate proxy
      python training.py --gbticl-model metalearner --coeff-model none \
          --lambda-rate 0.01 --epochs 30 --out checkpoints/stageA.pt

  Stage B: freeze the Stage A graph model and train the coefficient model
      python training.py --gbticl-model metalearner --freeze-gbticl \
          --gbticl-checkpoint checkpoints/stageA.pt \
          --coeff-model hf_lora --epochs 10 --out checkpoints/stageB.pt

  Stage C: joint fine-tuning of both models at a low learning rate
      python training.py --gbticl-model metalearner --coeff-model hf_lora \
          --gbticl-checkpoint checkpoints/stageA.pt \
          --coeff-checkpoint checkpoints/stageB.pt \
          --lr 3e-5 --epochs 5 --out checkpoints/stageC.pt

--gbticl-model net trains the GBTICLNet baseline (context-only MLP, no support set).
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise SystemExit(f"training.py requires PyTorch (import error: {e})")

from PIL import Image

from gbticl_pipeline.device_utils import get_device, load_state_dict_relaxed
from gbticl_pipeline.graph_model import (
    GBTICLNet, GBTICLMetaLearner, context_features, reference_edge_weights_batch,
)
from gbticl_pipeline.graph_utils import build_laplacian_batch, eigendecompose
from gbticl_pipeline.gft import forward_gft_batch, inverse_gft_batch
from gbticl_pipeline.coeff_model import TinyTransformerCoeffModel, HFLoRACoeffModel
from gbticl_pipeline.context import N_SUPPORT, get_block, get_context, get_support_set

__all__ = ["N_SUPPORT", "train_step_metalearner"]

BLOCK_SIZE = 8
PAD_VALUE = 128


# ---------------------------------------------------------------------------
# Datasets. The context of a training block is taken from the true frame; at
# inference the decoder's reconstruction approximates it (exactly so as the
# quantisation step tends to zero).
# ---------------------------------------------------------------------------
class BlockContextDataset(Dataset):
    """Randomly sampled (context, block) pairs for training GBTICLNet."""

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


class MetaEpisodeDataset(Dataset):
    """
    Episodes for meta-training GBTICLMetaLearner: a query block plus its support set.

    Contexts and support sets are built with the codec's own get_context /
    get_support_set / reference_edge_weights, from the true frames. Frame t uses
    frame t - 1 of the same sequence as the temporal support; the first frame of
    a sequence has none, as in encode_video. All frames are held in memory.
    """

    def __init__(self, sequences, block_size=8, samples_per_frame=1000, seed=0):
        """
        Args:
            sequences: dict mapping sequence name -> list of frame paths in time order
        """
        self.block_size = block_size
        self.frames = {}  # sequence name -> list of (H, W, 3) uint8 frames in time order
        self.items = []  # (sequence name, frame index, block row, block column)
        rng = np.random.default_rng(seed)

        for seq_name, paths in sequences.items():
            imgs = [np.array(Image.open(p).convert("RGB")) for p in paths]
            self.frames[seq_name] = imgs
            for t, img in enumerate(imgs):
                h, w = img.shape[:2]
                n_bh, n_bw = h // block_size, w // block_size
                all_idx = [(i, j) for i in range(n_bh) for j in range(n_bw)]
                n_take = min(samples_per_frame, len(all_idx))
                chosen = rng.choice(len(all_idx), size=n_take, replace=False)
                for c in chosen:
                    i, j = all_idx[c]
                    self.items.append((seq_name, t, i, j))

        n_frames = sum(len(v) for v in self.frames.values())
        print(f"MetaEpisodeDataset: {len(self.items)} sampled episodes across "
              f"{n_frames} frames, {len(sequences)} sequence(s)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq_name, t, i, j = self.items[idx]
        img = self.frames[seq_name][t]
        prev_img = self.frames[seq_name][t - 1] if t > 0 else None

        canvas = torch.from_numpy(img)
        prev_canvas = torch.from_numpy(prev_img) if prev_img is not None else None

        top, left, valid_top, valid_left = get_context(canvas, i, j, self.block_size)
        block = get_block(canvas, i, j, self.block_size)
        support = get_support_set(canvas, prev_canvas, i, j, self.block_size)

        return dict(
            top=top, left=left,
            valid_top=torch.tensor(valid_top), valid_left=torch.tensor(valid_left),
            block=block,
            support_top=support["top"], support_left=support["left"],
            support_valid_top=support["valid_top"], support_valid_left=support["valid_left"],
            support_block=support["block"], support_valid=support["valid"],
        )


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------
def train_step(gbticl_net, coeff_net, batch, quant_step, symbol_range, lambda_rate, device):
    """One rate-distortion step for GBTICLNet; returns (loss, distortion, rate in bits)."""
    top, left, valid_top, valid_left, block = batch
    top, left, block = top.to(device), left.to(device), block.to(device)
    valid_top, valid_left = valid_top.to(device), valid_left.to(device)

    # Graph weights -> Laplacian -> eigenbasis
    feats = gbticl_net.features_batch(top, left, valid_top, valid_left, device)
    logits_w = gbticl_net.forward(feats)
    weights = gbticl_net.to_weights(logits_w)  # (B, n_edges)

    L = build_laplacian_batch(weights, gbticl_net.block_size)  # (B, n, n)
    eigvals, U = eigendecompose(L)  # (B,n), (B,n,n)

    # Transform and quantise with a straight-through estimator
    coeffs = forward_gft_batch(block, U)  # (B, n, 3) float64

    coeffs_scaled = coeffs / quant_step
    q_hard = torch.round(coeffs_scaled)
    q_ste = coeffs_scaled + (q_hard - coeffs_scaled).detach()  # forward: rounded, backward: identity

    # Distortion
    recon = inverse_gft_batch(q_ste * quant_step, U, gbticl_net.block_size)  # (B, bs, bs, 3)
    distortion = F.mse_loss(recon, block.to(torch.float64))

    # Rate: cross-entropy of the coefficient model
    lo, hi = symbol_range
    n_symbols = hi - lo + 1
    b = q_hard.shape[0]

    q_int = torch.clamp(q_hard, lo, hi).to(torch.int64)  # (B, n, 3)
    target_idx = (q_int - lo)  # (B, n, 3), class indices

    # The three channels of a block share one spectrum but have separate coefficient
    # sequences, so (block, channel) is flattened into one batch axis; row r of every
    # *_rep tensor refers to the same (block, channel).
    eigvals_rep = eigvals.unsqueeze(1).expand(-1, 3, -1).reshape(b * 3, -1)  # (B*3, n)
    true_vals_rep = q_int.permute(0, 2, 1).reshape(b * 3, -1).to(torch.float32)  # (B*3, n)
    target_rep = target_idx.permute(0, 2, 1).reshape(b * 3, -1)  # (B*3, n)

    coeff_logits = coeff_net.forward_sequence(eigvals_rep, true_vals_rep)  # (B*3, n, n_symbols)
    ce_nats = F.cross_entropy(
        coeff_logits.reshape(-1, n_symbols), target_rep.reshape(-1), reduction="mean"
    )
    rate_bits = ce_nats / np.log(2.0)  # nats -> bits

    loss = distortion + lambda_rate * rate_bits
    return loss, distortion.item(), rate_bits.item()


def train_step_metalearner(gbticl_net, coeff_net, batch, quant_step, symbol_range, lambda_rate, device):
    """
    One episodic rate-distortion step for GBTICLMetaLearner.

    Same loss as train_step; only the graph prediction differs (cross-attention
    over the support set of each episode).

    coeff_net: trainable coefficient model for the rate term (Stage B/C), or None
        for Stage A, where the rate is the negative log-likelihood under the
        closed-form LaplaceCoeffModel (still differentiable w.r.t. the eigenvalues).
    """
    top = batch["top"].to(device)
    left = batch["left"].to(device)
    valid_top = batch["valid_top"].to(device)
    valid_left = batch["valid_left"].to(device)
    block = batch["block"].to(device)
    support_top = batch["support_top"].to(device)
    support_left = batch["support_left"].to(device)
    support_valid_top = batch["support_valid_top"].to(device)
    support_valid_left = batch["support_valid_left"].to(device)
    support_block = batch["support_block"].to(device)
    support_valid = batch["support_valid"].to(device)

    bs = gbticl_net.block_size
    b, k = support_top.shape[0], support_top.shape[1]

    # Query features and support examples (context features + reference graphs)
    query_feats = context_features(top, left, valid_top, valid_left, bs, device)  # (B, ctx_dim)
    support_ctx_feats = context_features(
        support_top.reshape(b * k, bs, 3), support_left.reshape(b * k, bs, 3),
        support_valid_top.reshape(b * k), support_valid_left.reshape(b * k), bs, device,
    ).reshape(b, k, -1)
    support_weights = reference_edge_weights_batch(
        support_block.reshape(b * k, bs, bs, 3), bs
    ).reshape(b, k, -1)

    # Predicted graph -> Laplacian -> eigenbasis
    logits_w = gbticl_net.forward(query_feats, support_ctx_feats, support_weights, support_valid)
    weights = gbticl_net.to_weights(logits_w)  # (B, n_edges)

    L = build_laplacian_batch(weights, bs)  # (B, n, n)
    eigvals, U = eigendecompose(L)  # (B,n), (B,n,n)
    coeffs = forward_gft_batch(block, U)  # (B, n, 3) float64

    # Straight-through quantisation and distortion
    coeffs_scaled = coeffs / quant_step
    q_hard = torch.round(coeffs_scaled)
    q_ste = coeffs_scaled + (q_hard - coeffs_scaled).detach()

    recon = inverse_gft_batch(q_ste * quant_step, U, bs)
    distortion = F.mse_loss(recon, block.to(torch.float64))

    lo, hi = symbol_range
    n_symbols = hi - lo + 1
    n = eigvals.shape[-1]
    q_int = torch.clamp(q_hard, lo, hi).to(torch.int64)  # (B, n, 3)
    target_idx = q_int - lo

    # Rate: learned model, or the Laplace proxy in Stage A
    if coeff_net is not None:
        eigvals_rep = eigvals.unsqueeze(1).expand(-1, 3, -1).reshape(b * 3, n)
        true_vals_rep = q_int.permute(0, 2, 1).reshape(b * 3, n).to(torch.float32)
        target_rep = target_idx.permute(0, 2, 1).reshape(b * 3, n)
        coeff_logits = coeff_net.forward_sequence(eigvals_rep, true_vals_rep)  # (B*3, n, n_symbols)
        ce_nats = F.cross_entropy(
            coeff_logits.reshape(-1, n_symbols), target_rep.reshape(-1), reduction="mean"
        )
        rate_bits = ce_nats / np.log(2.0)
    else:
        lap = train_step_metalearner._lap_cache
        if lap is None or lap._device_anchor.device != device:
            from gbticl_pipeline.coeff_model import LaplaceCoeffModel
            lap = LaplaceCoeffModel().to(device)
            train_step_metalearner._lap_cache = lap
        eigvals_flat = eigvals.unsqueeze(1).expand(-1, 3, -1).reshape(-1)  # (B*3*n,)
        probs_flat = lap.batch_symbol_probs(eigvals_flat, symbol_range)  # (B*3*n, n_symbols)
        probs = probs_flat.reshape(b * 3, n, n_symbols)
        target_rep = target_idx.permute(0, 2, 1).reshape(b * 3, n).clamp(0, n_symbols - 1)
        target_probs = torch.gather(probs, -1, target_rep.unsqueeze(-1)).squeeze(-1)
        nll_nats = -torch.log(target_probs.clamp_min(1e-12)).mean()
        rate_bits = nll_nats / np.log(2.0)

    loss = distortion + lambda_rate * rate_bits
    return loss, distortion.item(), rate_bits.item()


train_step_metalearner._lap_cache = None  # LaplaceCoeffModel, built on first use in Stage A


def _group_by_sequence(frame_paths):
    """Group frame paths by sequence (data/<Sequence>/frames/frameNNNN.png), each sorted by frame index."""
    groups = {}
    for p in frame_paths:
        seq_name = p.parent.parent.name if p.parent.name == "frames" else p.parent.name
        groups.setdefault(seq_name, []).append(p)
    for seq_name in groups:
        groups[seq_name] = sorted(groups[seq_name])
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dirs", nargs="+", default=["data/Beauty/frames", "data/HoneyBee/frames"],
                     help="One or more directories of extracted PNG frames.")
    ap.add_argument("--gbticl-model", choices=["metalearner", "net"], default="metalearner",
                     help="metalearner: GBTICLMetaLearner (episodic training with spatio-temporal "
                          "support sets). net: GBTICLNet baseline (context-only MLP).")
    ap.add_argument("--coeff-model", choices=["tiny", "hf_lora", "none"], default="tiny",
                     help="tiny: TinyTransformerCoeffModel (from scratch). hf_lora: HFLoRACoeffModel "
                          "(pretrained LM + LoRA). none: Laplace rate proxy, Stage A only "
                          "(requires --gbticl-model metalearner).")
    ap.add_argument("--base-model-name", type=str, default="distilgpt2",
                     help="Base pretrained LM for --coeff-model hf_lora.")
    ap.add_argument("--freeze-gbticl", action="store_true",
                     help="Freeze GBT-ICL (loaded from --gbticl-checkpoint) and train only the "
                          "coefficient model (Stage B).")
    ap.add_argument("--gbticl-checkpoint", type=str, default=None,
                     help="Initialise GBT-ICL from this checkpoint (required with --freeze-gbticl).")
    ap.add_argument("--coeff-checkpoint", type=str, default=None,
                     help="Initialise the coefficient model from this checkpoint.")
    ap.add_argument("--samples-per-frame", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                     help="Gradient-norm clip.")
    ap.add_argument("--quant-step", type=float, default=8.0)
    ap.add_argument("--symbol-lo", type=int, default=-2200)
    ap.add_argument("--symbol-hi", type=int, default=2200)
    ap.add_argument("--lambda-rate", type=float, default=0.01,
                     help="Weight of the rate term; larger values give smaller files and lower quality.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/gbticl_ckpt.pt")
    ap.add_argument("--num-workers", type=int, default=0,
                     help="DataLoader worker processes (0 = main process).")
    ap.add_argument("--log-every", type=int, default=20,
                     help="Print the running loss every N batches.")
    args = ap.parse_args()

    if args.coeff_model == "none" and args.gbticl_model != "metalearner":
        raise SystemExit("--coeff-model none is only valid with --gbticl-model metalearner (Stage A)")
    if args.freeze_gbticl and not args.gbticl_checkpoint:
        raise SystemExit("--freeze-gbticl requires --gbticl-checkpoint")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"training on device: {device}  gbticl-model={args.gbticl_model}  coeff-model={args.coeff_model}"
          f"{'  [gbticl FROZEN]' if args.freeze_gbticl else ''}")

    script_dir = Path(__file__).resolve().parent
    frame_paths = []
    for d in args.frames_dirs:
        d_path = (script_dir / d) if not Path(d).is_absolute() else Path(d)
        frame_paths += sorted(d_path.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"No PNG frames found under {args.frames_dirs} (resolved against {script_dir})")
    print(f"found {len(frame_paths)} frames")

    symbol_range = (args.symbol_lo, args.symbol_hi)

    # Graph model (optionally restored from a checkpoint)
    if args.gbticl_model == "metalearner":
        gbticl_net = GBTICLMetaLearner(block_size=BLOCK_SIZE).to(device)
    else:
        gbticl_net = GBTICLNet(block_size=BLOCK_SIZE).to(device)

    if args.gbticl_checkpoint:
        ckpt = torch.load(script_dir / args.gbticl_checkpoint, map_location=device, weights_only=False)
        load_state_dict_relaxed(gbticl_net, ckpt["gbticl_net"], args.gbticl_model)
        print(f"loaded GBT-ICL weights from {args.gbticl_checkpoint} (epoch {ckpt.get('epoch', '?')})")

    if args.freeze_gbticl:
        gbticl_net.eval()
        for p in gbticl_net.parameters():
            p.requires_grad_(False)

    # Coefficient model (optionally restored from a checkpoint)
    if args.coeff_model == "tiny":
        coeff_net = TinyTransformerCoeffModel(block_size=BLOCK_SIZE, symbol_range=symbol_range).to(device)
    elif args.coeff_model == "hf_lora":
        coeff_net = HFLoRACoeffModel(
            base_model_name=args.base_model_name, block_size=BLOCK_SIZE, symbol_range=symbol_range,
        ).to(device)
    else:
        coeff_net = None

    if coeff_net is not None and args.coeff_checkpoint:
        ckpt = torch.load(script_dir / args.coeff_checkpoint, map_location=device, weights_only=False)
        load_state_dict_relaxed(coeff_net, ckpt["coeff_net"], args.coeff_model)
        print(f"loaded coefficient-model weights from {args.coeff_checkpoint}")

    # Dataset and loader
    if args.gbticl_model == "metalearner":
        sequences = _group_by_sequence(frame_paths)
        print(f"grouped into {len(sequences)} sequence(s): "
              f"{', '.join(f'{k} ({len(v)} frames)' for k, v in sequences.items())}")
        dataset = MetaEpisodeDataset(sequences, BLOCK_SIZE, args.samples_per_frame, args.seed)
    else:
        dataset = BlockContextDataset(frame_paths, BLOCK_SIZE, args.samples_per_frame, args.seed)

    steps_per_epoch = len(dataset) // args.batch_size
    print(f"{steps_per_epoch} optimizer steps/epoch x {args.epochs} epochs = "
          f"{steps_per_epoch*args.epochs} total steps")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    params = [p for p in gbticl_net.parameters() if p.requires_grad]
    if coeff_net is not None:
        params += [p for p in coeff_net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    gbticl_label = type(gbticl_net).__name__
    coeff_label = type(coeff_net).__name__ if coeff_net is not None else "none (Stage A rate proxy)"
    print(f"total TRAINABLE parameters: {n_params:,} "
          f"({gbticl_label}: {sum(p.numel() for p in gbticl_net.parameters() if p.requires_grad):,}, "
          f"{coeff_label}: "
          f"{sum(p.numel() for p in coeff_net.parameters() if p.requires_grad) if coeff_net is not None else 0:,})")
    if not params:
        raise SystemExit("no trainable parameters -- check --freeze-gbticl / --coeff-model combination")

    optimizer = torch.optim.Adam(params, lr=args.lr)
    step_fn = train_step_metalearner if args.gbticl_model == "metalearner" else train_step

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
            loss, dist, rate = step_fn(
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

        ckpt_out = {
            "gbticl_net": gbticl_net.state_dict(),
            "gbticl_model_type": args.gbticl_model,
            "block_size": BLOCK_SIZE,
            "symbol_range": symbol_range,
            "quant_step": args.quant_step,
            "lambda_rate": args.lambda_rate,
            "epoch": epoch + 1,
            "history": history,
        }
        if coeff_net is not None:
            ckpt_out["coeff_net"] = coeff_net.state_dict()
            ckpt_out["coeff_model_type"] = args.coeff_model
            if args.coeff_model == "hf_lora":
                ckpt_out["base_model_name"] = args.base_model_name
        # The checkpoint is rewritten after every epoch
        torch.save(ckpt_out, out_path)
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
