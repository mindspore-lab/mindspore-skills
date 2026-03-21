#!/usr/bin/env python3
"""
Analyze code changes and extract operator information.

This module analyzes parsed diff data to identify operators, change types,
risk areas, and test coverage for code review purposes.
"""

import re
from typing import Dict, List, Set


def extract_operator_names(parsed_diff: Dict, framework: str) -> List[str]:
    """
    Extract operator names from file paths and code content.

    Args:
        parsed_diff: Output from parse_diff()
        framework: 'torch_npu' or 'mindspore'

    Returns:
        List of detected operator names
    """
    operators = set()

    for file_info in parsed_diff['files']:
        path = file_info['path']

        # Extract from file path
        if framework == 'mindspore':
            # mindspore/ops/operations/nn_ops.py -> nn_ops
            # mindspore/ops/op_def/yaml/add.yaml -> add
            if '/ops/' in path:
                name_match = re.search(r'/([a-z_]+)\.(?:py|yaml|cpp|cc)$', path)
                if name_match:
                    operators.add(name_match.group(1))

        elif framework == 'torch_npu':
            # torch_npu/csrc/aten/ops/AddKernel.cpp -> Add
            # torch_npu/csrc/aten/NPUNativeFunctions.cpp -> (multiple ops)
            if '/ops/' in path or 'Kernel' in path:
                name_match = re.search(r'/([A-Z][a-zA-Z0-9_]+)(?:Kernel)?\.(?:cpp|cu|h)$', path)
                if name_match:
                    operators.add(name_match.group(1))

        # Extract from code content (class/function names)
        for hunk in file_info['hunks']:
            content = hunk['content']
            # Look for operator definitions
            if framework == 'mindspore':
                # class Add(Primitive): or def add(...)
                op_matches = re.findall(r'class\s+([A-Z][a-zA-Z0-9]+)\s*\(', content)
                operators.update(op_matches)
            elif framework == 'torch_npu':
                # TORCH_LIBRARY_IMPL or at::Tensor add_npu(...)
                op_matches = re.findall(r'(?:TORCH_LIBRARY_IMPL|at::Tensor)\s+([a-z_]+)', content)
                operators.update(op_matches)

    return sorted(list(operators))


def identify_change_types(parsed_diff: Dict) -> List[str]:
    """
    Identify types of changes in the diff.

    Args:
        parsed_diff: Output from parse_diff()

    Returns:
        List of change types: 'forward', 'backward', 'shape_infer', 'dtype', etc.
    """
    change_types = set()

    for file_info in parsed_diff['files']:
        path = file_info['path']

        # Identify by file path
        if 'bprop' in path.lower() or 'backward' in path.lower():
            change_types.add('backward')
        if 'infer' in path.lower() or 'shape' in path.lower():
            change_types.add('shape_infer')
        if 'kernel' in path.lower() or '/ops/' in path:
            change_types.add('forward')
        if 'test' in path.lower():
            change_types.add('test')
        if path.endswith('.md') or 'doc' in path.lower():
            change_types.add('documentation')

        # Identify by code content
        for hunk in file_info['hunks']:
            content = hunk['content']
            if re.search(r'dtype|DataType|ScalarType', content, re.IGNORECASE):
                change_types.add('dtype')
            if re.search(r'shape|broadcast|reshape', content, re.IGNORECASE):
                change_types.add('shape_infer')
            if re.search(r'grad|backward|bprop', content, re.IGNORECASE):
                change_types.add('backward')

    return sorted(list(change_types))


def detect_risk_areas(parsed_diff: Dict) -> List[Dict]:
    """
    Detect potential risk areas in code changes.

    Args:
        parsed_diff: Output from parse_diff()

    Returns:
        List of risk areas with file, line, type, and description
    """
    risks = []

    for file_info in parsed_diff['files']:
        path = file_info['path']
        file_type = file_info['type']

        # Only analyze code files
        if file_type not in ['cpp', 'cuda', 'python', 'header']:
            continue

        for hunk in file_info['hunks']:
            content = hunk['content']
            lines = content.split('\n')
            line_num = hunk['new_start']

            for line in lines:
                # Skip deletion lines
                if line.startswith('-'):
                    continue
                if line.startswith('+'):
                    line_num += 1
                    code = line[1:]  # Remove '+' prefix

                    # Memory safety risks (C++/CUDA)
                    if file_type in ['cpp', 'cuda', 'header']:
                        if re.search(r'\*\s*\w+\s*=', code):  # Pointer assignment
                            risks.append({
                                'file': path,
                                'line': line_num,
                                'type': 'memory_safety',
                                'description': 'Pointer operation detected, ensure null check'
                            })
                        if re.search(r'\[\s*\w+\s*\]', code):  # Array access
                            risks.append({
                                'file': path,
                                'line': line_num,
                                'type': 'memory_safety',
                                'description': 'Array access detected, ensure bounds check'
                            })
                        if 'malloc' in code or 'new ' in code:
                            risks.append({
                                'file': path,
                                'line': line_num,
                                'type': 'memory_safety',
                                'description': 'Memory allocation detected, ensure proper cleanup'
                            })

                    # Numerical stability risks
                    if re.search(r'(?:log|exp|sqrt|pow)\s*\(', code, re.IGNORECASE):
                        risks.append({
                            'file': path,
                            'line': line_num,
                            'type': 'numerical_stability',
                            'description': 'Math operation detected, check for overflow/underflow'
                        })
                    if '/' in code and file_type == 'python':
                        risks.append({
                            'file': path,
                            'line': line_num,
                            'type': 'numerical_stability',
                            'description': 'Division detected, ensure zero-division handling'
                        })

                    # Shape inference risks
                    if re.search(r'shape\[|\.shape|broadcast', code, re.IGNORECASE):
                        risks.append({
                            'file': path,
                            'line': line_num,
                            'type': 'shape_inference',
                            'description': 'Shape operation detected, verify broadcasting rules'
                        })

                    # Dtype handling risks
                    if re.search(r'astype|cast|dtype|to\(', code):
                        risks.append({
                            'file': path,
                            'line': line_num,
                            'type': 'dtype_handling',
                            'description': 'Type conversion detected, verify dtype compatibility'
                        })

                elif not line.startswith('-'):
                    line_num += 1

    return risks


def analyze_test_coverage(parsed_diff: Dict) -> Dict:
    """
    Analyze test coverage in the diff.

    Args:
        parsed_diff: Output from parse_diff()

    Returns:
        Dictionary with has_tests, test_files, and missing_tests
    """
    test_files = []
    impl_files = []

    for file_info in parsed_diff['files']:
        if file_info['type'] == 'test':
            test_files.append(file_info['path'])
        elif file_info['type'] in ['python', 'cpp', 'cuda']:
            impl_files.append(file_info['path'])

    has_tests = len(test_files) > 0

    # Suggest missing test scenarios
    missing_tests = []
    if impl_files and not has_tests:
        missing_tests.append('Basic functionality test')
        missing_tests.append('Edge cases (empty input, zero dimensions)')
        missing_tests.append('Different dtypes (float32, float16, int32)')
        missing_tests.append('Different shapes (1D, 2D, high-dimensional)')
        missing_tests.append('Gradient correctness test')

    return {
        'has_tests': has_tests,
        'test_files': test_files,
        'missing_tests': missing_tests
    }


def analyze_operator_changes(parsed_diff: Dict, framework: str) -> Dict:
    """
    Analyze operator changes in the diff.

    Args:
        parsed_diff: Output from parse_diff()
        framework: 'torch_npu' or 'mindspore'

    Returns:
        Dictionary with operators, change_types, risk_areas, and test_coverage
    """
    return {
        'operators': extract_operator_names(parsed_diff, framework),
        'change_types': identify_change_types(parsed_diff),
        'risk_areas': detect_risk_areas(parsed_diff),
        'test_coverage': analyze_test_coverage(parsed_diff)
    }


if __name__ == '__main__':
    import sys
    import json
    from parse_diff import parse_diff

    if len(sys.argv) < 3:
        print("Usage: python3 analyze_changes.py <diff_file> <framework>")
        print("Framework: 'torch_npu' or 'mindspore'")
        sys.exit(1)

    diff_file = sys.argv[1]
    framework = sys.argv[2]

    with open(diff_file, 'r', encoding='utf-8') as f:
        diff_content = f.read()

    parsed_diff = parse_diff(diff_content)
    analysis = analyze_operator_changes(parsed_diff, framework)

    print(json.dumps(analysis, indent=2, ensure_ascii=False))
