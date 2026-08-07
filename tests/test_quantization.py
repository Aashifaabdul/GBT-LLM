import torch

from gbticl_pipeline.quantization import quantize, dequantize


def test_quantize_dequantize_roundtrip_within_half_step():
    torch.manual_seed(0)
    coeffs = (torch.rand(1000, dtype=torch.float64) - 0.5) * 4096
    step = 8.0
    q = quantize(coeffs, step)
    dq = dequantize(q, step)
    assert torch.all((dq - coeffs).abs() <= step / 2 + 1e-9)


def test_quantize_dtype_is_integral():
    coeffs = torch.tensor([1.4, -1.4, 100.6], dtype=torch.float64)
    q = quantize(coeffs, 1.0)
    assert q.dtype == torch.int64


def test_quantize_zero_step_free_passthrough_semantics():
    # step -> small approaches lossless: with a small enough step, dequantize
    # should recover the original to within that step's resolution
    coeffs = torch.tensor([3.3, -7.7, 0.0], dtype=torch.float64)
    step = 0.01
    q = quantize(coeffs, step)
    dq = dequantize(q, step)
    assert torch.allclose(dq, coeffs, atol=step)
