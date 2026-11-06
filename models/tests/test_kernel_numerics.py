"""Compile actual graph kernels and compare their simulator results with PyTorch."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SIM = Path(os.environ.get('ATALLA_SIM_ROOT', ROOT / 'atalla-sim'))
sys.path[:0] = [str(ROOT), str(SIM / 'src')]
import build_model
from validation_metrics import compare_cosine_max_mean
from scheduler.elf import load_elf
from atalla.atalla_platform import build_atalla_platform, c_entry_sregs


def read_tensor(platform, code, name, shape):
    match = re.search(r'static TileDesc32 tensor_' + re.escape(name) + r'_tiles\[\d+\] = \{(.*?)\n\};', code, re.S)
    assert match, name
    addresses = [int(value, 16) for value in re.findall(r'base_addr = 0x([0-9A-F]+)', match.group(1))]
    logical_shape = tuple(shape)
    shape = (1, shape[0]) if len(shape) == 1 else tuple(shape)
    height, width = shape[-2:]
    planes = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    out = np.empty((planes, height, width), dtype=np.float32)
    index = 0
    for plane in range(planes):
        for row in range(0, height, 32):
            for col in range(0, width, 32):
                raw = platform.dram.read(addresses[index], 2048)
                tile = (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).view('<f4').reshape(32, 32)
                h, w = min(32, height-row), min(32, width-col)
                out[plane, row:row+h, col:col+w] = tile[:h, :w]
                index += 1
    return out.reshape(logical_shape)


def run_case(model, x, directory, *, limit=500_000, atol=0.003, rtol=0.02):
    model = copy.deepcopy(model).bfloat16().eval()
    x = x.bfloat16()
    graph = build_model.build_graph(model, x)
    output = next(node for node in graph.graph.nodes if node.op == 'output').args[0]
    with patch.object(build_model, 'load_model', return_value=(model, x)):
        elf = build_model.build('basic', directory, 42, .01)
    program = load_elf(elf)
    platform = build_atalla_platform(program.instr, program.data,
        init_sregs=c_entry_sregs(program.data), describe_pc=program.describe, strict=True)
    platform.run_until_done(limit=limit)
    assert platform.core.decode2.srf.storage.values[10] == 0, 'graph kernel returned failure'
    with torch.no_grad():
        expected = model(x).float().numpy()
    actual = read_tensor(platform, (directory / 'graph_schedule.c').read_text(), output.name, expected.shape)
    cosine, _, _ = compare_cosine_max_mean(actual, expected)
    if np.linalg.norm(expected) > 0:
        assert cosine > 0.99, f"cosine similarity too low: {cosine:.6f}"
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return platform


class BiasAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('bias', torch.linspace(-1, 1, 32))
    def forward(self, x):
        return x + self.bias


def test_bias_add(tmp_path):
    run_case(BiasAdd(), torch.linspace(-2, 2, 32).reshape(1, 32), tmp_path, atol=0.016)


class MatrixMultiply(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', ((torch.arange(32*32).reshape(32, 32) % 11)-5).float() / 16)
    def forward(self, x):
        return torch.matmul(x, self.weight)


def test_matmul(tmp_path):
    run_case(MatrixMultiply(), ((torch.arange(32) % 7)-3).float().reshape(1,32)/8, tmp_path)


def test_relu(tmp_path):
    run_case(nn.ReLU(), torch.linspace(-1, 1, 32).reshape(1,32), tmp_path, atol=0, rtol=0)


def test_basic_model(tmp_path):
    torch.manual_seed(42)
    model, x = build_model.load_model('basic')
    run_case(model, x, tmp_path, atol=0.006, rtol=0.03)


class TailMatrixMultiply(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(123)
        self.register_buffer('weight', torch.randn(35, 37)/16)
    def forward(self, x):
        return torch.matmul(x, self.weight)


def test_matmul_multiple_tiles_and_tails(tmp_path):
    torch.manual_seed(124)
    run_case(TailMatrixMultiply(), torch.randn(3,35)/4, tmp_path, atol=0.003, rtol=0.03)


class ScalarMultiply(nn.Module):
    def forward(self, x):
        return x * 0.5


def test_scalar_multiply(tmp_path):
    run_case(ScalarMultiply(), torch.linspace(-1,1,32).reshape(1,32), tmp_path, atol=0, rtol=0)


def test_convolution(tmp_path):
    model = nn.Sequential(nn.Conv2d(1, 2, 1, bias=True))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[[[0.5]]], [[[-0.25]]]]))
        model[0].bias.copy_(torch.tensor([0.125, -0.125]))
    run_case(model, torch.tensor([[[[0.5, -0.5], [1., -1.]]]]), tmp_path, limit=1_000_000)


def test_maxpool(tmp_path):
    model = nn.Sequential(nn.MaxPool2d(2,2))
    run_case(model, torch.tensor([[[[-1.,2.,3.,-4.], [5.,-6.,7.,8.], [9.,10.,-11.,12.], [-13.,14.,15.,16.]]]]), tmp_path, atol=0, rtol=0)


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


def test_flatten(tmp_path):
    run_case(Flatten(), torch.arange(32).float().reshape(1,2,4,4), tmp_path, atol=0, rtol=0)


def test_spatial_graph(tmp_path):
    torch.manual_seed(99)
    model = nn.Sequential(nn.Conv2d(2,3,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(1),nn.Linear(12,5))
    # Use the functional flatten lowering exposed by the runner.
    class Graph(nn.Module):
        def __init__(self, parts):
            super().__init__()
            self.conv=parts[0];self.fc=parts[4]
        def forward(self,x):
            x=torch.nn.functional.max_pool2d(torch.relu(self.conv(x)),2,2)
            return self.fc(torch.flatten(x,1))
    run_case(Graph(model),torch.randn(1,2,4,4)/4,tmp_path,limit=1_000_000,atol=.01,rtol=.03)


def test_convolution_reduction_tiles(tmp_path):
    torch.manual_seed(125)
    model=nn.Sequential(nn.Conv2d(4,2,3))
    run_case(model,torch.randn(1,4,3,3)/4,tmp_path,atol=.005,rtol=.03)


def test_grouped_convolution_without_bias(tmp_path):
    torch.manual_seed(126)
    model=nn.Sequential(nn.Conv2d(2,2,1,groups=2,bias=False))
    run_case(model,torch.randn(1,2,2,2)/4,tmp_path)


class TensorMultiply(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('rhs',torch.linspace(-1,1,66).reshape(2,33))
    def forward(self,x):return x*self.rhs


def test_tensor_multiply_tails(tmp_path):
    run_case(TensorMultiply(),torch.linspace(-2,2,66).reshape(2,33),tmp_path,atol=0,rtol=0)


class ScaledBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('bias',torch.linspace(-1,1,35))
    def forward(self,x):return torch.add(x,self.bias,alpha=-0.5)


def test_scaled_bias_across_tiles(tmp_path):
    run_case(ScaledBias(),torch.linspace(-2,2,33*35).reshape(33,35),tmp_path,atol=.016,rtol=0)


def test_runtime_rhs_layout_is_rejected_before_link(tmp_path):
    class RuntimeWeights(nn.Module):
        def forward(self,x):return torch.matmul(x,x+x)
    model=RuntimeWeights()
    x=torch.ones(32,32)
    with patch.object(build_model,'load_model',return_value=(model,x)):
        with pytest.raises(RuntimeError,match='unsupported operations'):
            build_model.build('basic',tmp_path,42,.01)
    manifest=__import__('json').loads((tmp_path/'kernel_manifest.json').read_text())
    assert not manifest['kernel_coverage']
    assert any(not binding['storage_compatible'] for binding in manifest['bindings'])
    assert not (tmp_path/'output.elf').exists()


def test_convolution_batches_channel_tiles_stride_and_dilation(tmp_path):
    torch.manual_seed(127)
    model=nn.Sequential(nn.Conv2d(2,33,2,stride=2,padding=1,dilation=2))
    run_case(model,torch.randn(2,2,3,3)/4,tmp_path,limit=1_000_000,atol=.005,rtol=.03)


def test_convolution_multiple_spatial_blocks(tmp_path):
    model=nn.Sequential(nn.Conv2d(1,1,1,bias=False))
    with torch.no_grad():model[0].weight.fill_(0.5)
    run_case(model,torch.arange(36).reshape(1,1,6,6).float()/16,tmp_path,atol=0,rtol=0)
