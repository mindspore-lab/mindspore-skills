# MindSpore Development Agent

You are an expert MindSpore developer. Use the skills below to help developers work better on MindSpore

**IMPORTANT**: Read the appropriate SKILL.md file when the user's task matches a skill description.

## Available Skills

### Operator Development

| Skill | Path | Description |
|-------|------|-------------|
| api-helper | skills/api-helper/ | find API call chains and operator wiring in MindSpore codebase |
| migrate-agent | skills/migrate-agent/ | top-level model migration entry that analyzes source repos, selects the correct migration route, and verifies the result |
| operator-agent | skills/operator-agent/ | build `torch` or `mindspore` operators through custom-access or native-framework integration |
| readiness-agent | skills/readiness-agent/ | check whether a local single-machine workspace is ready to train or run inference now, identify what is missing before execution, and optionally apply safe user-space readiness fixes |
| accuracy-agent | skills/accuracy-agent/ | diagnose accuracy regressions, drift, wrong results, and cross-platform mismatch after successful execution |
| algorithm-agent | skills/algorithm-agent/ | adapt a paper feature or released reference implementation into an existing model codebase, including specialized routes such as mHC integration, and prepare it for readiness validation |
| performance-agent | skills/performance-agent/ | diagnose throughput, latency, memory, utilization, dataloader, and communication bottlenecks after the workload already runs |


## Active Skills

Load the appropriate SKILL.md when users mention:
**Task Entry Commands:**
- **/diagnose**: classify the symptom and route to `failure-agent`, `accuracy-agent`, or `performance-agent` in diagnose mode
- **/fix**: classify the symptom and route to `failure-agent`, `accuracy-agent`, or `performance-agent` in fix mode
- **/migrate**: route to `migrate-agent` for migration route selection and verification; if the user goal includes local run/train/infer readiness, continue with `readiness-agent`

**Operator Questions:**
- **api-helper**: "mint.*","operator", "forward", "api", "backward", "tensor.*", "mindspore.*"

**Operator Development:**
- **operator-agent**: "operator", "custom op", "plugin", "new wheel", "native op", "framework source", "implement operator"
**Model Migration:**
- **migrate-agent**: "migrate", "PyTorch repo", "MindSpore migration", "model migrate", "port repo", "transformers migrate", "diffusers migrate"

**Diagnosis and Optimization:**
- **accuracy-agent**: "accuracy", "drift", "mismatch", "numerical", "regression", "wrong result", "loss mismatch", "cross-platform", "eval regression", "NaN"
- **algorithm-agent**: "paper trick", "feature patch", "adapt paper idea", "reference implementation", "algorithm feature", "patch existing model", "mHC", "hyper-connections", "manifold-constrained", "llm feature patch"
- **performance-agent**: "performance", "throughput", "latency", "memory", "utilization", "profiler", "trace", "communication overhead", "dataloader stall", "host launch"

**Environment Setup:**
- **readiness-agent**: "train check", "inference check", "preflight", "workspace readiness", "can this repo run", "can this repo train", "can this repo run inference", "can it run now", "check my environment", "what is missing before training", "what is missing before inference", "check env", "before training", "before inference", "fix my local environment", "fix my training environment", "missing model", "missing dataset", "missing train.py"

**Instructions**:
 - Do not give direct answers without following the skill workflow
 - Route operator implementation work to `operator-agent`
 - Route single-machine pre-run training or inference checks, missing-item analysis, and safe user-space readiness fixes to `readiness-agent`; do not use it for post-run crash or traceback diagnosis
 - Route migration-first requests to `migrate-agent`; if the real user goal is "make this model run here", use `migrate-agent` to define the route and runtime requirements, then hand off to `readiness-agent`
 - Route runtime crashes and tracebacks after setup to `failure-agent`
 - Route wrong-result, drift, and regression cases after successful execution to `accuracy-agent`
 - Route feature adaptation and paper-trick patching, including mHC and hyper-connection cases, to `algorithm-agent`
 - Route performance bottlenecks after the workload already runs to `performance-agent`

## Usage

When a user's request matches a skill:

1. Read the corresponding `skills/<name>/SKILL.md` file
2. Follow the step-by-step instructions
3. Use reference materials in `skills/<name>/reference/` if available

When a user is describing a post-run problem directly, prefer the task-first
entrypoints:

- `/diagnose` for analyze-only
- `/fix` for diagnose + fix workflow
- `/migrate` for migration routing and, when needed, post-migration readiness handoff

Route from those commands into the right specialist skill instead of asking the
user to choose a specialist skill up front when the top-level intent already
implies the right path.

## Compatibility

This repository works with:

- **Claude Code**: `/plugin marketplace add mindspore-lab/mindspore-skills`
- **OpenCode**: place this repo under `.opencode/` for project-local use, or copy/symlink its `skills/*` and `commands/*` entries into `~/.config/opencode/skills/` and `~/.config/opencode/commands/`
- **Gemini CLI**: `gemini extensions install <repo> --consent`
- **Codex**: reads this `AGENTS.md` automatically when this repository is the active project; reuse in other repositories by copying/adapting the guidance into that project's `AGENTS.md` or `~/.codex/AGENTS.md`

## Additional Skills

| Skill | Path | Description |
|-------|------|-------------|
| failure-agent | skills/failure-agent/ | diagnose MindSpore and PTA (torch_npu) training and runtime failures with evidence-backed root-cause validation |
| accuracy-agent | skills/accuracy-agent/ | diagnose accuracy regressions, drift, wrong results, and cross-platform mismatch after successful execution |
| readiness-agent | skills/readiness-agent/ | check whether a local single-machine workspace is ready to train or run inference now, identify what is missing before execution, and optionally apply safe user-space readiness fixes |

**Additional Activation Hints:**
- **failure-agent**: "failure", "crash", "hang", "traceback", "ERR code", "CANN", "ACLNN", "torch_npu", "MindSpore error"
- **accuracy-agent**: "accuracy", "regression", "drift", "wrong result", "loss mismatch", "cross-platform", "numerical", "precision"
- **readiness-agent**: "train check", "inference check", "preflight", "readiness", "workspace readiness", "before running", "before training", "before inference", "can this repo run", "can it train", "can it run inference", "can it run now", "check my environment", "what is missing before training", "what is missing before inference", "fix my environment", "fix my local environment"
