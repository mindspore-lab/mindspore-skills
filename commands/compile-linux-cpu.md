---
name: compile-linux-cpu
description: Linux x86_64 CPU compilation workflow
skill: compile-linux-cpu
---

# Compile MindSpore on Linux CPU

This command invokes the `compile-linux-cpu` skill for compiling MindSpore from source on Linux x86_64 with CPU support.

## Usage

```
/compile-linux-cpu
```

The skill will guide you through:
1. Setting up conda environment
2. Installing system dependencies (Ubuntu/Debian or CentOS/RHEL)
3. Installing compiler (GCC 7.3+ or Clang 9.0+)
4. Preparing source code
5. Installing build dependencies
6. Compiling MindSpore
7. Installing and verifying the build

## Prerequisites

- Linux x86_64 (Ubuntu 18.04+, CentOS 7+, Debian 10+)
- GCC 7.3+ or Clang 9.0+
- Python 3.9-3.12
- 20GB disk space
- sudo access for system packages

## See Also

- `/compile-macos` - For macOS Apple Silicon compilation
