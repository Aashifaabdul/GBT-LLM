"""context.py::get_support_set -- spatial + temporal support-set assembly
for GBTICLMetaLearner, and graph_model.py::reference_edge_weights."""

import torch

from gbticl_pipeline.context import get_support_set, N_SUPPORT, get_context, get_block
from gbticl_pipeline.graph_model import edge_list, reference_edge_weights, reference_edge_weights_batch


def test_support_set_shape_and_dtypes(device):
    canvas = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8, device=device)
    prev_canvas = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8, device=device)
    support = get_support_set(canvas, prev_canvas, 2, 2, block_size=8)

    assert support["top"].shape == (N_SUPPORT, 8, 3)
    assert support["left"].shape == (N_SUPPORT, 8, 3)
    assert support["block"].shape == (N_SUPPORT, 8, 8, 3)
    assert support["valid_top"].shape == (N_SUPPORT,)
    assert support["valid_left"].shape == (N_SUPPORT,)
    assert support["valid"].shape == (N_SUPPORT,)
    assert support["top"].dtype == torch.uint8
    assert support["valid"].dtype == torch.bool


def test_support_set_top_left_corner_all_missing(device):
    """Block (0,0) has no spatial neighbours (nothing decoded before it in
    raster order) and, with no previous frame, no temporal neighbours
    either -- every one of the 8 support slots must be marked invalid."""
    canvas = torch.zeros((32, 32, 3), dtype=torch.uint8, device=device)
    support = get_support_set(canvas, None, 0, 0, block_size=8)
    assert not support["valid"].any()


def test_support_set_interior_block_matches_get_context(device):
    """Spatial support slots should return exactly the same context/block
    data as calling get_context/get_block directly at that position --
    get_support_set must not reimplement this differently."""
    canvas = torch.arange(32 * 32 * 3, dtype=torch.uint8, device=device).reshape(32, 32, 3) % 256
    support = get_support_set(canvas, None, 2, 2, block_size=8)

    # slot 0 = left neighbour (0,-1) -> block (2,1)
    top_expected, left_expected, vt, vl = get_context(canvas, 2, 1, 8)
    assert torch.equal(support["top"][0], top_expected)
    assert torch.equal(support["left"][0], left_expected)
    assert support["valid_top"][0] == vt
    assert support["valid_left"][0] == vl
    assert torch.equal(support["block"][0], get_block(canvas, 2, 1, 8))
    assert support["valid"][0].item() is True


def test_support_set_temporal_disabled_without_prev_frame(device):
    canvas = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8, device=device)
    support = get_support_set(canvas, None, 2, 2, block_size=8)
    assert not support["valid"][4:].any()


def test_reference_edge_weights_uniform_block_gives_uniform_weights(device):
    """A perfectly flat (constant-colour) block has zero pixel differences
    everywhere -> every edge weight should be exp(0) = 1."""
    block = torch.full((8, 8, 3), 150, dtype=torch.uint8, device=device)
    weights = reference_edge_weights(block, block_size=8)
    n_edges = len(edge_list(8))
    assert weights.shape == (n_edges,)
    assert torch.allclose(weights, torch.ones(n_edges, dtype=torch.float64, device=device))


def test_reference_edge_weights_sharp_edge_gets_downweighted(device):
    """A block with a hard vertical edge (left half dark, right half bright)
    should down-weight the horizontal edges crossing that boundary relative
    to edges within a flat region."""
    block = torch.zeros((8, 8, 3), dtype=torch.uint8, device=device)
    block[:, :4, :] = 10
    block[:, 4:, :] = 240
    weights = reference_edge_weights(block, block_size=8)

    edges = edge_list(8)
    # find a horizontal edge crossing the boundary (col 3 -> col 4) vs one
    # entirely within the dark region (col 0 -> col 1)
    crossing_idx = edges.index((0 * 8 + 3, 0 * 8 + 4))
    flat_idx = edges.index((0 * 8 + 0, 0 * 8 + 1))
    assert weights[crossing_idx] < weights[flat_idx]


def test_reference_edge_weights_batch_matches_single(device):
    torch.manual_seed(0)
    blocks = torch.randint(0, 256, (5, 8, 8, 3), dtype=torch.uint8, device=device)
    batch_weights = reference_edge_weights_batch(blocks, block_size=8)
    for b in range(5):
        single = reference_edge_weights(blocks[b], block_size=8)
        assert torch.allclose(batch_weights[b], single)
