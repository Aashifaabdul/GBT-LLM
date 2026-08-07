"""training.py's episodic meta-training step, exercised with synthetic
in-memory data (no dependency on the real extracted-frame dataset being
present) -- catches tensor-shape contract bugs like the one found during
development (HFLoRACoeffModel.forward_sequence not supporting a real
(B, n) batch, only used previously with a bare (n,) sequence)."""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train_step_metalearner, N_SUPPORT  # noqa: E402
from gbticl_pipeline.graph_model import GBTICLMetaLearner  # noqa: E402
from gbticl_pipeline.coeff_model import TinyTransformerCoeffModel  # noqa: E402

WIDE_RANGE = (-2200, 2200)
BLOCK_SIZE = 8


def _synthetic_batch(b, device):
    n = BLOCK_SIZE
    return dict(
        top=torch.randint(0, 256, (b, n, 3), dtype=torch.uint8, device=device),
        left=torch.randint(0, 256, (b, n, 3), dtype=torch.uint8, device=device),
        valid_top=torch.randint(0, 2, (b,), dtype=torch.bool, device=device),
        valid_left=torch.randint(0, 2, (b,), dtype=torch.bool, device=device),
        block=torch.randint(0, 256, (b, n, n, 3), dtype=torch.uint8, device=device),
        support_top=torch.randint(0, 256, (b, N_SUPPORT, n, 3), dtype=torch.uint8, device=device),
        support_left=torch.randint(0, 256, (b, N_SUPPORT, n, 3), dtype=torch.uint8, device=device),
        support_valid_top=torch.randint(0, 2, (b, N_SUPPORT), dtype=torch.bool, device=device),
        support_valid_left=torch.randint(0, 2, (b, N_SUPPORT), dtype=torch.bool, device=device),
        support_block=torch.randint(0, 256, (b, N_SUPPORT, n, n, 3), dtype=torch.uint8, device=device),
        support_valid=torch.randint(0, 2, (b, N_SUPPORT), dtype=torch.bool, device=device),
    )


def test_train_step_metalearner_stage_a_no_coeff_model(device):
    """Stage A: GBT-ICL alone, coeff_net=None -- uses the LaplaceCoeffModel
    closed-form rate proxy internally."""
    gbticl_net = GBTICLMetaLearner(block_size=BLOCK_SIZE).to(device)
    batch = _synthetic_batch(4, device)

    loss, dist, rate = train_step_metalearner(
        gbticl_net, None, batch, quant_step=8.0, symbol_range=WIDE_RANGE, lambda_rate=0.01, device=device,
    )
    assert torch.isfinite(loss)
    assert dist == dist and rate == rate  # not NaN

    loss.backward()
    grads = [p.grad for p in gbticl_net.parameters()]
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_train_step_metalearner_stage_c_joint(device):
    """Stage C-style joint step: both GBT-ICL and a trainable coeff model
    receive gradients."""
    gbticl_net = GBTICLMetaLearner(block_size=BLOCK_SIZE).to(device)
    coeff_net = TinyTransformerCoeffModel(block_size=BLOCK_SIZE, symbol_range=WIDE_RANGE).to(device)
    batch = _synthetic_batch(4, device)

    loss, dist, rate = train_step_metalearner(
        gbticl_net, coeff_net, batch, quant_step=8.0, symbol_range=WIDE_RANGE, lambda_rate=0.01, device=device,
    )
    assert torch.isfinite(loss)

    loss.backward()
    gbticl_grads = [p.grad for p in gbticl_net.parameters()]
    coeff_grads = [p.grad for p in coeff_net.parameters()]
    assert any(g is not None and g.abs().sum().item() > 0 for g in gbticl_grads)
    assert any(g is not None and g.abs().sum().item() > 0 for g in coeff_grads)


def test_train_step_metalearner_stage_b_frozen_gbticl(device):
    """Stage B-style step: GBT-ICL frozen (requires_grad=False), only the
    coefficient model's gradients should be nonzero."""
    gbticl_net = GBTICLMetaLearner(block_size=BLOCK_SIZE).to(device)
    for p in gbticl_net.parameters():
        p.requires_grad_(False)
    coeff_net = TinyTransformerCoeffModel(block_size=BLOCK_SIZE, symbol_range=WIDE_RANGE).to(device)
    batch = _synthetic_batch(4, device)

    loss, dist, rate = train_step_metalearner(
        gbticl_net, coeff_net, batch, quant_step=8.0, symbol_range=WIDE_RANGE, lambda_rate=0.01, device=device,
    )
    loss.backward()
    assert all(p.grad is None for p in gbticl_net.parameters())
    coeff_grads = [p.grad for p in coeff_net.parameters()]
    assert any(g is not None and g.abs().sum().item() > 0 for g in coeff_grads)
