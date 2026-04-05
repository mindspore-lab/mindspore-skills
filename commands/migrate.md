---
description: Route to MindSpore migration tools (HF models, third-party repos)
---

# Migrate

Use this as the top-level migration entrypoint.

Route internally to `migrate-agent` after classifying the source:

- Hugging Face Transformers model or repo
- Hugging Face Diffusers pipeline or repo
- generic PyTorch repository

If the user only wants a migration plan or a code port, stop at
`migrate-agent`.

If the user says they want to run, train, infer, set up, or otherwise make the
migrated model runnable on the current machine, do this:

1. run `migrate-agent` first
2. extract the runtime requirements and expected runnable artifacts
3. hand off to `readiness-agent` for local readiness checking or safe user-space
   preparation

Examples of local runnable intent:

- `I want to run qwen3 in MindSpore`
- `Port this repo and make it train here`
- `Migrate this transformers model and check whether it can run now`

## Usage

```text
/migrate migrate this Hugging Face Transformers Qwen3 model to MindSpore
/migrate port this PyTorch repo to MindSpore
/migrate I want to run this Hugging Face Transformers Qwen3 model in MindSpore on this machine
```

If the user does not specify what they want to migrate, ask for the source repo
or model family before routing.
