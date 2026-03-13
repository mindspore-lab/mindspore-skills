# MindSpore Skills

MindSpore development skills for AI coding agents. Build CPU/GPU/NPU operators and migrate models with guided workflows.

Compatible with **Claude Code**, **OpenCode**, **Gemini CLI**, and **Codex**.

## Installation

### Claude Code

Register the marketplace and install:

```
/plugin marketplace add vigo999/mindspore-skills
/plugin install mscode@mindspore-skills
```

Then use slash command:

```
/mscode:cpu-plugin-builder
/mscode:cpu-native-builder
/mscode:gpu-builder
/mscode:hf-diffusers-migrate
/mscode:hf-migrate
/mscode:hf-transformers-migrate
/mscode:hf-transformers-migrate-test
/mscode:migrate
/mscode:model-migrate
/mscode:npu-builder
/mscode:mindspore-aclnn-operator-devflow
/mscode:api-helper
/mscode:compile-linux-cpu
/mscode:compile-macos
```

### OpenCode

Clone to your global config:

```bash
git clone https://github.com/vigo999/mindspore-skills.git ~/.config/opencode/mindspore-skills
ln -s ~/.config/opencode/mindspore-skills/skills ~/.config/opencode/skills
ln -s ~/.config/opencode/mindspore-skills/commands ~/.config/opencode/commands
```

Or for a specific project:

```bash
git clone https://github.com/vigo999/mindspore-skills.git .opencode
```

Then in OpenCode:

```
/cpu-plugin-builder
```

See [OpenCode Skills docs](https://opencode.ai/docs/skills) for more details.

### Gemini CLI

Install as an extension:

```bash
gemini extensions install https://github.com/vigo999/mindspore-skills.git --consent
```

Or from local clone:

```bash
git clone https://github.com/vigo999/mindspore-skills.git
gemini extensions install ./mindspore-skills --consent
```

See [Gemini CLI extensions docs](https://geminicli.com/docs/extensions/) for more details.

### Codex

Clone to your project root:

```bash
git clone https://github.com/vigo999/mindspore-skills.git .mindspore-skills
```

Codex reads `AGENTS.md` automatically. Verify with:

```bash
codex "Summarize the MindSpore skills available."
```

See [Codex AGENTS guide](https://developers.openai.com/codex/guides/agents-md) for more details.

## Available Skills

### Operator Development

| Skill | Description |
|-------|-------------|
| `api-helper` | Find API call chains and operator wiring in MindSpore codebase |
| `cpu-plugin-builder` | Build CPU operators via ATen/libtorch adaptation |
| `cpu-native-builder` | Build native CPU kernels with Eigen/SLEEF |
| `gpu-builder` | Build GPU operators with CUDA |
| `npu-builder` | Build NPU operators for Huawei Ascend |
| `mindspore-aclnn-operator-devflow` | End-to-end ACLNN operator adaptation workflow for MindSpore Ascend |

### Model Migration

| Skill | Description |
|-------|-------------|
| `hf-diffusers-migrate` | Migrate HF diffusers models to mindone.diffusers |
| `hf-transformers-migrate` | Migrate Hugging Face transformers models to mindone.transformers |
| `hf-transformers-migrate-test` | Generate minimal MindOne transformer tests for migrated models |
| `model-migrate` | Migrate PyTorch repos to MindSpore |

### Compilation

| Skill | Description |
|-------|-------------|
| `compile-linux-cpu` | Compile MindSpore from source on Linux x86_64 CPU |
| `compile-macos` | Compile MindSpore from source on macOS Apple Silicon |

## Available Commands

### Operator Development

| Command | Description |
|---------|-------------|
| `/api-helper` | API chain discovery workflow |
| `/cpu-plugin-builder` | ATen adaptation workflow |
| `/cpu-native-builder` | Native kernel workflow |
| `/gpu-builder` | CUDA kernel workflow |
| `/npu-builder` | Ascend NPU workflow |
| `/mindspore-aclnn-operator-devflow` | ACLNN deep development workflow |

### Model Migration

| Command | Description |
|---------|-------------|
| `/migrate` | Migration router (HF/third-party), routing only |
| `/hf-migrate` | HF library router (diffusers/transformers), routing only |
| `/hf-diffusers-migrate` | HF diffusers migration workflow |
| `/hf-transformers-migrate` | HF transformers migration workflow |
| `/hf-transformers-migrate-test` | HF transformers test generation for migrated models |
| `/model-migrate` | PyTorch repo migration workflow |

### Compilation

| Command | Description |
|---------|-------------|
| `/compile-linux-cpu` | Linux x86_64 CPU compilation workflow |
| `/compile-macos` | macOS Apple Silicon compilation workflow |

## Usage Examples

### Build a CPU operator

```
/cpu-plugin-builder

> Help me implement the linspace operator for MindSpore CPU
```

### Run minimal public example

```bash
bash examples/cpu/plugin_add/run.sh
```

See expected result contract in `examples/cpu/plugin_add/expected.md`.

## Contract and Tests

- Contract docs:
  - `docs/concepts/skills-contract.md`
  - `docs/concepts/artifacts-and-reporting.md`
- Cross-skill contract tests: `tests/contract/`
- Skill-specific tests: `skills/<skill>/tests/`

## Repository Structure

```
mindspore-skills/
├── .claude-plugin/          # Claude Code plugin config
├── commands/                # Slash commands
│   ├── api-helper.md        # API chain discovery
│   ├── migrate.md           # Migration router
│   ├── hf-migrate.md        # HF library router
│   └── ...
├── skills/                  # Skill definitions
│   ├── cpu-plugin-builder/  # ATen/libtorch operators
│   ├── cpu-native-builder/  # Native CPU kernels
│   ├── gpu-builder/         # CUDA operators
│   ├── npu-builder/         # Ascend NPU operators
│   ├── mindspore-aclnn-operator-devflow/ # ACLNN deep development workflow
│   ├── hf-diffusers-migrate/   # HF diffusers migration
│   ├── hf-transformers-migrate/ # HF transformers migration
│   ├── hf-transformers-migrate-test/ # HF transformers test generation after migration
│   └── model-migrate/       # PyTorch repo migration
├── AGENTS.md                # Codex instructions
└── gemini-extension.json    # Gemini CLI config
```

## Contributing

Contributions are welcome. Please submit a pull request.

When adding a new skill:
1. Add `skills/<skill-name>/SKILL.md` with matching frontmatter and directory name
2. Add a slash command in `commands/<command-name>.md` if needed
3. Update `AGENTS.md` (skill table + activation triggers)
4. Update `README.md` (skill list and commands)
5. Update `gemini-extension.json` with name/path/description
6. Update `.claude-plugin/marketplace.json` if skills or metadata changed

When modifying an existing skill:
1. Update `skills/<skill-name>/SKILL.md` and any referenced files
2. Refresh `AGENTS.md` triggers if scope/keywords changed
3. Update `README.md` if descriptions or commands changed
4. Update `gemini-extension.json` if name/path/description changed

tools:
- Run `python tools/check_consistency.py` before submit
- (Optional) Install git hooks with `python tools/install_git_hooks.py`
- Tip: set up hooks with `make hooks` (see Makefile).

## License

Apache 2.0
