# AI Skills for MindSpore and torch_npu

A curated collection of production-oriented AI skills for operator debugging, code review,
web content capture, and MHTML-to-Markdown conversion.

This repository is designed for engineering workflows around MindSpore, torch_npu, and
Ascend ecosystem development, with practical scripts and references included in each skill.

[中文文档 (Chinese README)](./README.zh-CN.md)

## What's Inside

- **Operator debugging workflow** for MindSpore ops issues (analysis -> scoping -> fixing -> validation)
- **Operator code review workflow** for torch_npu and MindSpore diffs/PRs
- **Web capture workflow** with Playwright for authenticated pages
- **MHTML conversion workflow** with optional OCR support via Ollama

## Skill Index

| Skill | Purpose | Typical Trigger |
|---|---|---|
| [`mindspore-ops-debugger`](./mindspore-ops-debugger/SKILL.md) | End-to-end diagnosis and fix guidance for MindSpore operator issues | Kernel error, shape inference issue, precision mismatch, bprop/gradient anomaly, ACLNN adaptation |
| [`ascend-ops-reviewer`](./ascend-ops-reviewer/SKILL.md) | Structured review for torch_npu/MindSpore operator changes | "review PR", local diff review, operator code audit |
| [`web-fetch`](./web-fetch/SKILL.md) | Capture webpages as MHTML/PDF/PNG/HTML with login support | Fetch protected pages, batch archive URLs, save web evidence |
| [`mhtml-to-md`](./mhtml-to-md/SKILL.md) | Convert `.mhtml` archives to clean Markdown, with OCR for images | Convert web archives to text docs, extract issue content |

## Repository Layout

```text
skills/
├── mindspore-ops-debugger/
├── ascend-ops-reviewer/
├── web-fetch/
└── mhtml-to-md/
```

Most skill folders include:
- `SKILL.md`: trigger rules and workflow instructions
- `references/`: checklists, patterns, and troubleshooting guides
- `scripts/`: helper automation for parsing/fetching/conversion tasks

## Quick Start

### 1) Clone this repository

```bash
git clone <your-repo-url> skills
cd skills
```

### 2) Use a skill directly

Open the target `SKILL.md` and follow its workflow in your AI coding assistant.

Examples:
- MindSpore ops bug investigation: `mindspore-ops-debugger/SKILL.md`
- Diff-based operator review: `ascend-ops-reviewer/SKILL.md`
- Authenticated webpage capture: `web-fetch/SKILL.md`
- MHTML to Markdown conversion: `mhtml-to-md/SKILL.md`

### 3) Install optional runtime dependencies (per skill)

Some skills include executable scripts and need extra runtime tools:
- `web-fetch`: Python deps + Playwright browser
- `mhtml-to-md`: Python deps + Ollama (`glm-ocr:bf16`) for OCR mode

Check each skill's `SKILL.md` for exact commands and options.

## Who This Repo Is For

- MindSpore / torch_npu operator developers
- AI infra engineers doing bug triage and regression analysis
- Teams that archive web issue pages and convert them to technical notes
- Engineers building repeatable, tool-assisted debugging/review pipelines

## Contributing

Contributions are welcome. Recommended scope:
- Improve workflow clarity in `SKILL.md`
- Add new references and real-world case patterns
- Extend helper scripts and examples
- Improve bilingual documentation quality

Please keep updates practical, reproducible, and aligned with real debugging/review scenarios.

## Language

- English: `README.md` (this file)
- Chinese: `README.zh-CN.md`
