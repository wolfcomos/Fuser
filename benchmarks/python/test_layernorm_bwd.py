# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import (
    run_benchmark,
    clear_dynamo_cache,
    unary_bwd_torch,
    with_executor,
    DEFAULT_EXECUTORS,
)
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES
import numpy as np
from .torch_ops import layernorm


def layernorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )

    T2 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    T3 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    )

    T4 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.cast(T4, dtype=DataType.Float)

    V8 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T9 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
    V12 = T0.shape()
    T13 = fd.ops.broadcast_in_dim(T9, shape=V12, broadcast_dims=[0, 1])
    T14 = fd.ops.sub(T0, T13)

    T18 = fd.ops.broadcast_in_dim(T3, shape=V12, broadcast_dims=[0, 1])
    T19 = fd.ops.mul(T14, T18)

    T23 = fd.ops.broadcast_in_dim(T4, shape=V12, broadcast_dims=[1])
    T28 = fd.ops.sum(T1, dims=[0], keepdim=False, dtype=DataType.Null)

    T30 = fd.ops.mul(T1, T23)
    T31 = fd.ops.mul(T1, T19)
    T32 = fd.ops.sum(T31, dims=[0], keepdim=False, dtype=DataType.Null)

    T34 = fd.ops.mul(T30, T18)
    T35 = fd.ops.mul(T30, T14)
    T36 = fd.ops.sum(T35, dims=[1], keepdim=False, dtype=DataType.Null)

    T40 = fd.ops.broadcast_in_dim(T36, shape=V8, broadcast_dims=[0])
    T41 = fd.ops.neg(T34)
    T42 = fd.ops.sum(T41, dims=[1], keepdim=False, dtype=DataType.Null)
    T46 = fd.ops.broadcast_in_dim(T42, shape=V8, broadcast_dims=[0])
    S47 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T48 = fd.ops.mul(S47, T40)
    S49 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T50 = fd.ops.pow(T3, S49)
    T51 = fd.ops.mul(T48, T50)
    T54 = fd.ops.sum(T46, dims=[1], keepdim=False, dtype=DataType.Null)
    T55 = fd.ops.sum(T51, dims=[1], keepdim=False, dtype=DataType.Null)

    T59 = fd.ops.broadcast_in_dim(T55, shape=V8, broadcast_dims=[0])
    T63 = fd.ops.broadcast_in_dim(T59, shape=V12, broadcast_dims=[0, 1])
    T67 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
    T71 = fd.ops.broadcast_in_dim(T67, shape=V12, broadcast_dims=[0, 1])

    S72 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T73 = fd.ops.mul(S72, T63)
    T74 = fd.ops.sub(T0, T71)
    T75 = fd.ops.mul(T73, T74)

    S77 = fd.ops.reciprocal(T0.size(1))
    T78 = fd.ops.mul(T75, S77)
    T82 = fd.ops.broadcast_in_dim(T54, shape=V8, broadcast_dims=[0])
    T86 = fd.ops.broadcast_in_dim(T82, shape=V12, broadcast_dims=[0, 1])
    T88 = fd.ops.mul(S77, T86)
    T89 = fd.ops.add(T78, T88)
    T90 = fd.ops.add(T34, T89)

    if dtype in PROMOTE_DTYPES:
        T28 = fd.ops.cast(T28, dtype=dtype)
        T90 = fd.ops.cast(T90, dtype=dtype)
        T32 = fd.ops.cast(T32, dtype=dtype)

    fd.add_output(T90)
    fd.add_output(T32)
    fd.add_output(T28)


def layernorm_bwd_iobytes(size: tuple, dtype: torch.dtype):
    # Manual IOBytes computation since nvfuser input/outputs (in_tensor, grad_out, mean, invstd, weigts) differ from baselines (out, grad_out)
    # Total IO bytes = in_tensor (size, dtype) + grad_out (size, dtype) + mean (size[0], float) +
    #       invstd (size[0], float) + weights (size[1], dtype) +
    #       grad_in (size, dtype) + grad_weights (size[1], dtype)+ grad_bias (size[1], dtype)
    return int(
        dtype.itemsize * 3 * (np.prod(size) + size[1])
        + torch.float.itemsize * 2 * size[0]
    )


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.inner_outer_persistent
def test_layernorm_bwd_nvf_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    mean = inputs.to(torch.float).mean(dim=-1)
    variance = inputs.to(torch.float).var(dim=-1, unbiased=False)
    invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)

    with FusionDefinition() as fd:
        layernorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.nn.functional.layer_norm(
            inputs.to(torch.double),
            inputs.shape[1:],
            weight=weights.to(torch.double),
            bias=bias.to(torch.double),
        )
        eager_output.backward(grads.to(torch.double))
        fd.validate(
            [inputs, grads, mean, invstd, weights],
            [inputs.grad, weights.grad, bias.grad],
        )

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, grads, mean, invstd, weights])


@pytest.mark.parametrize("executor", DEFAULT_EXECUTORS)
@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layernorm_bwd_baseline_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()

    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    fwd_fn = with_executor(executor, layernorm)
    fwd_inputs = [inputs, weights, bias]
    outputs = fwd_fn(fwd_inputs)

    # Manually compute IOBytes: See PR #1725
    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs, grads, *fwd_inputs],
        iobytes=layernorm_bwd_iobytes(size, dtype),
    )
