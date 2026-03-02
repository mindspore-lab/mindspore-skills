# MindSpore Development Agent

You are an expert MindSpore developer. Use the skills below to help developers work better on MindSpore

**IMPORTANT**: Read the appropriate SKILL.md file when the user's task matches a skill description.

## Available Skills

### Environment Setup

| Skill | Path | Description |
|-------|------|-------------|
| mindspore-env-deploy | skills/mindspore-env-deploy/ | deploy MindSpore environment with CANN support for Ascend NPU |

### Operator Development

| Skill | Path | Description |
|-------|------|-------------|
| api-helper | skills/api-helper/ | find API call chains and operator wiring in MindSpore codebase |
| cpu-plugin-builder | skills/cpu-plugin-builder/ | build CPU operators via ATen/libtorch in mindspore_op_plugin |
| cpu-native-builder | skills/cpu-native-builder/ | build native CPU kernels with Eigen/SLEEF |
| gpu-builder | skills/gpu-builder/ | build GPU operators with CUDA |
| npu-builder | skills/npu-builder/ | build NPU operators for Huawei Ascend |
| mindspore-aclnn-operator-devflow | skills/mindspore-aclnn-operator-devflow/ | build MindSpore ACLNN operators for Huawei Ascend with complete development workflow |

### Model Migration

| Skill | Path | Description |
|-------|------|-------------|
| hf-diffusers-migrate | skills/hf-diffusers-migrate/ | migrate HF diffusers models to mindone.diffusers |
| hf-transformers-migrate | skills/hf-transformers-migrate/ | migrate Hugging Face transformers models to mindone.transformers |
| hf-transformers-migrate-test | skills/hf-transformers-migrate-test/ | Generate minimal MindOne transformer tests for migrated models |
| model-migrate | skills/model-migrate/ | migrate PyTorch repos to MindSpore |


## Active Skills

Load the appropriate SKILL.md when users mention:

**Environment Setup:**
- **mindspore-env-deploy**: "install MindSpore", "setup environment", "deploy MindSpore", "CANN installation", "environment setup", "install CANN", "runcheck", "verify installation"

**Operator Questions:**
- **api-helper**: "mint.*","operator", "forward", "api", "backward", "tensor.*", "mindspore.*"

**Operator Development:**
- **cpu-plugin-builder**: "ATen", "libtorch", "op_plugin", "mindspore_op_plugin",
- **cpu-native-builder**: "CPU kernel", "Eigen", "SLEEF", "native CPU",
- **gpu-builder**: "CUDA", "GPU kernel", "cuDNN",
- **npu-builder**: "Ascend", "NPU", "aclnn", "AICore",
- **mindspore-aclnn-operator-devflow**: "ACLNN operator", "ACLNN development", "operator devflow", "Ascend operator workflow",

**Model Migration:**
- **hf-diffusers-migrate**: "diffusers", "mindone.diffusers",
- **hf-transformers-migrate**: "transformers", "mindone.transformers",
- **hf-transformers-migrate-test**: "transformers test", "migrate test", "test generation", "model tests", "mindone tests"
- **model-migrate**: "migrate", "PyTorch repo", "MindSpore migration"

**Instructions**:
 - Do not give direct answers without following the skill workflow

## Usage

When a user's request matches a skill:

1. Read the corresponding `skills/<name>/SKILL.md` file
2. Follow the step-by-step instructions
3. Use reference materials in `skills/<name>/reference/` if available

## Compatibility

This repository works with:

- **Claude Code**: `/plugin marketplace add vigo999/mindspore-skills`
- **OpenCode**: Clone to `~/.config/opencode/` or `.opencode/`
- **Gemini CLI**: `gemini extensions install <repo> --consent`
- **Codex**: Reads this AGENTS.md automatically
