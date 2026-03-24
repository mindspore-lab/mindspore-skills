# Readiness-Agent Execution Target Discovery

This reference defines how `readiness-agent` should discover the intended
execution target.

Use it whenever the skill needs to answer:

- what the user is actually trying to run
- whether the target is `training` or `inference`
- which script, config, model, dataset, and checkpoint belong to that target

## Core Rule

Do not judge readiness before identifying the intended execution target.

Readiness is always for a specific execution path, not for the machine or
workspace in general.

## Evidence Order

Use evidence in this priority order:

1. explicit user input
2. launch scripts and config files
3. workspace code and imports
4. model and checkpoint markers
5. local environment facts

Rules:

- higher-priority evidence should dominate lower-priority evidence unless it is
  clearly contradicted
- do not infer the framework only from the model name
- do not infer the final target from a single weak marker

## Discovery Goals

The goal of this stage is to build an `ExecutionTarget`.

The `ExecutionTarget` should include, when available:

- `target_type`
- `entry_script`
- `launch_cmd`
- `framework_path`
- `config_path`
- `model_path`
- `dataset_path`
- `checkpoint_path`
- `output_path`
- `evidence`
- `confidence`

## Step 1. Explicit User Input

Treat explicit user input as the strongest signal.

Examples:

- "check whether `train.py` can run"
- "make sure this repo can do inference"
- "verify `scripts/finetune_qwen.sh` after my changes"

If the user provides:

- explicit script path
- explicit task type
- explicit config path
- explicit model path

use them as the starting target unless the workspace strongly contradicts them.

## Step 2. Launch Scripts And Config Files

Search for likely entrypoints such as:

- `train.py`
- `finetune.py`
- `infer.py`
- `generate.py`
- launch shell scripts
- notebooks that clearly represent the runnable path

Search for likely configs such as:

- yaml files
- json files
- shell-exported config references
- launcher arguments that point to config files

Prefer scripts or configs that:

- are referenced by other scripts
- appear in launcher commands
- live near model, dataset, or output definitions
- contain framework-specific training or inference settings

## Step 3. Workspace Code And Imports

Inspect the code to infer:

- framework path such as MindSpore or PTA
- training versus inference intent
- required runtime packages
- dataset or tokenizer usage
- whether resume or checkpoint-backed execution is intended

Useful signals include:

- trainer imports
- dataloader construction
- optimizer usage
- generation calls
- `set_context(device_target='Ascend')`
- `torch_npu` usage

## Step 4. Model And Checkpoint Markers

Use model and checkpoint markers only as supporting evidence, not as the sole
basis for target selection.

Useful markers include:

- `config.json`
- tokenizer files
- `model.safetensors`
- `*.ckpt`
- adapter or LoRA artifacts

These may help resolve:

- likely model path
- whether inference assets are present
- whether checkpoint resume is intended

## Step 5. Local Environment Facts

Local environment facts may confirm plausibility, but should not define the
target by themselves.

Examples:

- installed framework packages
- available Python environment
- importable runtime packages

Use them to validate a proposed target, not to invent one from nothing.

## Determining `training` vs `inference`

Choose `training` when evidence strongly indicates:

- train or finetune entry scripts
- optimizer or trainer setup
- dataset loading for learning
- loss computation
- checkpoint save / output training directory

Choose `inference` when evidence strongly indicates:

- infer or generate entry scripts
- model load and tokenizer load without trainer flow
- forward or generation path without training loop
- serving or prediction command shape

If strong evidence for both exists:

- choose the path with the strongest explicit execution evidence
- otherwise ask the user or emit `WARN`

## Multiple Candidate Targets

If multiple plausible targets exist:

- score them by evidence strength
- prefer explicit user intent
- prefer launcher-linked scripts over orphan scripts
- prefer the path with more complete config and asset linkage

Do not silently pick one if the ambiguity materially affects readiness.

If ambiguity remains:

- ask the user for clarification, or
- emit `WARN` and explain the ambiguity in the report

## Confidence Guidance

Confidence should reflect how strong and complete the evidence is.

Higher confidence:

- explicit user target
- launcher command references
- config-script-model linkage
- framework and asset evidence all align

Lower confidence:

- only file-name heuristics
- orphan scripts
- conflicting config and code signals
- model markers without a clear runnable path

## Output Requirements

The final discovered target should be stable enough to drive:

- dependency-closure discovery
- compatibility validation
- blocker classification
- target-specific smoke validation

If target discovery is not stable enough for later stages:

- do not silently continue as if certainty exists
- downgrade to `WARN` or request clarification

## Invariants

The skill must preserve these rules:

- final `target` must resolve to `training` or `inference`
- readiness is never certified for an unresolved target
- a weak model-name guess is never enough to claim a framework path
- ambiguous candidate scripts must not silently become a `READY` result
