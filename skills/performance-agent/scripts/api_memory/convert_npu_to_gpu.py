#!/usr/bin/env python3
"""
Convert an NPU memory test script to the equivalent GPU version.

Transformation rules (extracted from NPU/GPU benchmark script pairs):
  1. Remove import torch_npu / from torch_npu ...
  2. torch_npu.npu.xxx  -> torch.cuda.xxx
  3. torch.npu.xxx      -> torch.cuda.xxx
  4. .npu() tensor method -> .cuda()
  5. 'hccl' distributed backend -> 'nccl'
  6. 'npu' / 'npu:N'    -> 'cuda' / 'cuda:N'  (device strings, all contexts)
  7. ASCEND_RT_VISIBLE_DEVICES -> CUDA_VISIBLE_DEVICES
  8. pta_reserved_GB    -> gpu_reserved_GB
  9. pta_activated_GB   -> gpu_activated_GB
  10. NPU in comments/docstrings -> GPU

Usage:
    python convert_npu_to_gpu.py <npu_script_path> [-o <output_path>]

Examples:
    python convert_npu_to_gpu.py torchapi_id0299_nanmean.py
    python convert_npu_to_gpu.py torchapi_id4861_all_gather.py -o all_gather_gpu.py
"""

import argparse
import os
import re
import sys


TRANSFORMATIONS = [
    # Order matters: process longer/more specific patterns first to avoid
    # short patterns matching prematurely.

    # ── Remove torch_npu imports ──
    (r'^import torch_npu\s*\n', '', 'remove "import torch_npu"'),
    (r'^from torch_npu\b.*\n', '', 'remove "from torch_npu ..." import'),

    # ── Module/API path replacement (longer path first) ──
    (r'torch_npu\.npu\.', 'torch.cuda.', 'torch_npu.npu.* → torch.cuda.*'),
    (r'torch\.npu\.', 'torch.cuda.', 'torch.npu.* → torch.cuda.*'),

    # ── Tensor .npu() method -> .cuda() ──
    (r'\.npu\(\)', '.cuda()', '.npu() → .cuda()'),

    # ── Distributed backend hccl -> nccl ──
    (r"(['\"])hccl\1", r'\1nccl\1', "'hccl' → 'nccl'"),

    # ── Device string 'npu'/'npu:N' -> 'cuda'/'cuda:N' ──
    # Use (:\d+|) instead of (:\d+)? to ensure the group always participates,
    # avoiding back-reference anomalies.
    (r"(['\"])npu(:\d+|)\1", r'\1cuda\2\1', "'npu'/'npu:N' → 'cuda'/'cuda:N'"),

    # ── Ascend env var -> CUDA equivalent ──
    (r'\bASCEND_RT_VISIBLE_DEVICES\b', 'CUDA_VISIBLE_DEVICES',
     'ASCEND_RT_VISIBLE_DEVICES → CUDA_VISIBLE_DEVICES'),

    # ── Variable name / JSON key replacement ──
    (r'\bpta_reserved_GB\b', 'gpu_reserved_GB', 'pta_reserved_GB → gpu_reserved_GB'),
    (r'\bpta_activated_GB\b', 'gpu_activated_GB', 'pta_activated_GB → gpu_activated_GB'),

    # ── NPU -> GPU in comments/docstrings ──
    # \bNPU\b only matches the standalone uppercase word; will not alter
    # variable names like NPU_XXX or XXX_NPU.
    (r'\bNPU\b', 'GPU', 'NPU → GPU (in comments/docstrings)'),
]


def convert_npu_to_gpu(content):
    """Apply all transformation rules. Return (converted_content, applied_rules)."""
    applied = []
    result = content
    for pattern, replacement, desc in TRANSFORMATIONS:
        new_result = re.sub(pattern, replacement, result, flags=re.MULTILINE)
        if new_result != result:
            applied.append(desc)
            result = new_result
    return result, applied


def default_output_path(input_path):
    """input.py -> input_gpu.py; if _npu suffix exists, replace with _gpu."""
    base, ext = os.path.splitext(input_path)
    if base.endswith('_npu'):
        return base[:-4] + '_gpu' + ext
    return base + '_gpu' + ext


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPU memory test script to GPU version',
    )
    parser.add_argument('input', help='NPU test script path')
    parser.add_argument('-o', '--output', help='Output GPU script path (default: <input>_gpu.py)')
    parser.add_argument('--diff', action='store_true', help='Only show transformations that would be applied, do not write file')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'Error: file not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        content = f.read()

    converted, applied = convert_npu_to_gpu(content)

    if not applied:
        print('No NPU-specific patterns detected; please ensure the input file is an NPU script.')
        return

    print(f'Applied {len(applied)} transformation rule(s):')
    for rule in applied:
        print(f'  * {rule}')

    if args.diff:
        return

    output_path = args.output or default_output_path(args.input)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(converted)

    print(f'GPU script written: {output_path}')


if __name__ == '__main__':
    main()
