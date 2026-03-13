---
name: compile-macos
description: macOS Apple Silicon compilation workflow
skill: compile-macos
---

# Compile MindSpore on macOS

This command invokes the `compile-macos` skill for compiling MindSpore from source on macOS Apple Silicon.

## Usage

```
/compile-macos
```

The skill will guide you through:
1. Setting up conda environment
2. Preparing source code
3. Checking dependencies (Xcode tools, build tools, Python packages)
4. Compiling MindSpore
5. Installing and verifying the build

## Prerequisites

- macOS (Apple Silicon: M1/M2/M3)
- Apple Clang (via Xcode Command Line Tools)
- Python 3.9-3.12
- 20GB disk space

## See Also

- `/compile-linux-cpu` - For Linux x86_64 CPU compilation
