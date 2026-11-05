#!/usr/bin/env python3
"""Build and execute a complete model ELF, comparing each result with PyTorch."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch
import build_model
from validation_metrics import compare_cosine_max_mean


def read_tensor(platform, info):
    logical = tuple(info['shape'])
    shape = (1, logical[0]) if len(logical) == 1 else logical
    height, width = shape[-2:]
    planes = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    out = np.empty((planes,height,width), dtype=np.float32)
    address = info['address']
    for plane in range(planes):
        for y in range(0,height,32):
            for x in range(0,width,32):
                bits=np.frombuffer(platform.dram.read(address,2048),dtype='<u2').astype('<u4')
                tile=(bits<<16).view('<f4').reshape(32,32)
                h,w=min(32,height-y),min(32,width-x)
                out[plane,y:y+h,x:x+w]=tile[:h,:w]
                address+=2048
    return out.reshape(logical)


def verify(model_name, out_dir, sim_root, seed=42, scale=.01, limit=10_000_000, atol=.01, rtol=.03):
    elf=build_model.build(model_name,out_dir,seed,scale)
    sys.path.insert(0,str(sim_root/'src'))
    from scheduler.elf import load_elf
    from atalla.atalla_platform import build_atalla_platform,c_entry_sregs
    program=load_elf(elf)
    platform=build_atalla_platform(program.instr,program.data,
        init_sregs=c_entry_sregs(program.data),describe_pc=program.describe,strict=True)
    started=time.monotonic()
    while not platform.core.done:
        if platform.cycle>=limit:
            raise RuntimeError(f'No halt after {limit} cycles: {program.describe(platform.core.pc)}')
        platform.tick()
        if platform.cycle%100_000==0:
            print(f'Simulated {platform.cycle:,} cycles ({time.monotonic()-started:.1f}s): {program.describe(platform.core.pc)}',flush=True)
    status=platform.core.decode2.srf.storage.values[10]
    if status:
        raise RuntimeError(f'Graph returned nonzero status 0x{status:X}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    model,x=build_model.load_model(model_name,scale)
    graph=build_model.build_graph(model,x)
    from scripts.generate_schedule import _detect_scaled_args
    folded=_detect_scaled_args(graph)
    expected={}
    class Capture(torch.fx.Interpreter):
        def run_node(self,node):
            value=super().run_node(node)
            if isinstance(value,torch.Tensor):expected[node.name]=value.detach().float().numpy()
            return value
    with torch.no_grad():Capture(graph).run(x.bfloat16())
    tensors=json.loads((out_dir/'tensor_manifest.json').read_text())
    manifest=json.loads((out_dir/'kernel_manifest.json').read_text())
    nodes={n.name:n for n in graph.graph.nodes}
    results=[]
    for binding in manifest['bindings']:
        name=binding['node']
        if nodes[name] in folded:continue
        info=tensors[name]
        if info['storage_layout']!=0:continue
        actual=read_tensor(platform,info)
        reference=expected[name]
        cosine, max_error, mean_error = compare_cosine_max_mean(actual, reference)
        passed=bool(np.allclose(actual,reference,atol=atol,rtol=rtol,equal_nan=True))
        results.append(dict(node=name,operation=binding['operation'],passed=passed,
            cosine_similarity=cosine,max_abs=max_error,mean_abs=mean_error))
    report=dict(model=model_name,seed=seed,scale=scale,cycles=platform.cycle,status=status,
        elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
        atol=atol,rtol=rtol,passed=all(r['passed'] for r in results),results=results)
    (out_dir/'numerical_report.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest['numerical_validation']={key:report[key] for key in ('passed','cycles','atol','rtol','elf_sha256')}
    (out_dir/'kernel_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for r in results:print(f"{'PASS' if r['passed'] else 'FAIL'} {r['node']}: cos {r['cosine_similarity']:.6f}; max abs {r['max_abs']:.6g}; mean abs {r['mean_abs']:.6g}")
    print(f"{model_name}: {'PASS' if report['passed'] else 'FAIL'}; {platform.cycle:,} cycles; status {status}")
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('basic','alexnet_small'),default='basic')
    parser.add_argument('--out-dir',type=Path,default=Path('out/graph'))
    parser.add_argument('--sim-root',type=Path,default=Path(__file__).resolve().parent.parent/'atalla-sim')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--scale',type=float,default=.01)
    parser.add_argument('--limit',type=int,default=10_000_000)
    parser.add_argument('--atol',type=float,default=.01)
    parser.add_argument('--rtol',type=float,default=.03)
    args=parser.parse_args()
    report=verify(args.model,args.out_dir,args.sim_root,args.seed,args.scale,args.limit,args.atol,args.rtol)
    sys.exit(0 if report['passed'] else 1)
