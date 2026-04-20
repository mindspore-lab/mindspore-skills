# Structured Confirmation Design

## Purpose

Make `new-readiness-agent` ask users for configuration choices through a stable,
portable structured-confirmation contract that works well with host-native
question UIs such as `AskUserQuestion`, without hard-coding one agent platform.

## Problem

`new-readiness-agent` already has a strong per-field confirmation flow:

- the pipeline stops at `NEEDS_CONFIRMATION`
- it exposes `current_confirmation`
- it advances one field at a time through `--confirm field=value`

That flow is already the right product model. The instability comes from the
last mile:

- the skill text does not explicitly require a host-native structured question
  tool
- `current_confirmation.options` can contain more than four entries
- Claude Code and `mindspore-cli` both constrain `AskUserQuestion` to 2-4
  options per question

If we only strengthen prompt wording, the host model still has to improvise a
tool-friendly question from a larger option set. That is the unstable part.

## Goals

- Keep the current one-field-at-a-time confirmation workflow.
- Keep `current_confirmation` and `artifacts/confirmation-step.json` as the
  single source of truth.
- Add a portable structured-question projection that host runtimes can map to
  native tools such as `AskUserQuestion`.
- Avoid branching the skill into Claude Code only vs `mindspore-cli` only
  behavior.

## Non-Goals

- Do not add runtime-specific code paths in the skill itself.
- Do not replace the existing numbered full-option confirmation view.
- Do not require immediate `skill.yaml` schema changes. The shared skill schema
  currently rejects new top-level metadata, so this is deferred.

## Design

### 1. Single Source Of Truth

The canonical interaction state remains:

- `readiness-verdict.json.current_confirmation`
- `artifacts/confirmation-step.json`
- `readiness-output/latest/new-readiness-agent/confirmation-latest.json`

The host agent should read the current step from those artifacts instead of
reconstructing a question from raw workspace evidence.

### 2. Two Confirmation Views

Each confirmation step exposes two parallel views:

- `options`
  The full numbered confirmation view used by reports, debugging, and existing
  CLI/manual flows. This preserves all detected candidates plus `__manual__`
  and `__unknown__` where applicable.
- `portable_question`
  A host-tool-friendly projection for structured multiple-choice UIs.

`portable_question` contains:

- `header`
- `question`
- `multi_select`
- `options`
- `selection_strategy`
- `full_option_count`
- `response_binding`

Each portable option contains:

- `value`
- `label`
- `description`
- `recommended`
- `source_option_index`

### 3. Option Reduction Strategy

The portable question must always stay within the common denominator supported
by current host tools.

Rules:

- Keep one question per field.
- Keep `multi_select = false`.
- Never copy the explicit `__manual__` option into `portable_question`.
- Preserve `__unknown__` as a visible option when possible.
- If the portable projection would otherwise expose fewer than two visible
  options, add a manual-entry fallback option that tells the host to use its
  manual-input path.
- If the full option list is longer than four entries, shorten it to:
  - the recommended option first when one exists
  - the remaining highest-ranked candidates
  - `__unknown__`

This keeps the portable view compatible with `AskUserQuestion` while preserving
the full confirmation catalog in `options`.

### 4. Manual Entry Contract

Manual entry remains part of the step contract through:

- `allow_free_text`
- `manual_hint`
- the full `options` list

The portable question intentionally omits the explicit manual option in normal
cases because host-native question tools may already provide their own `Other`
path. If the portable projection would otherwise drop below two visible
choices, it may surface a manual-entry fallback option. If a host does not
provide built-in manual entry, it can fall back to the full step plus
`manual_hint`.

### 5. Resume Contract

The confirmation step must keep the existing continuation model:

- answer one field
- map the selected portable option `value` back to that field
- rerun the pipeline with `--confirm field=value`

The `response_binding` object in `portable_question` makes this explicit.

## Why This Design

- It aligns with the existing product model instead of creating a second
  confirmation system.
- It improves cross-platform stability because the host no longer has to invent
  a tool-shaped question from raw candidate lists.
- It keeps current reports and artifacts useful for humans because the full
  numbered options still exist.
- It keeps future runtime work small because host agents only need to map
  `portable_question` into their native UI.

## Implementation Phases

### Phase 1

- Add this design document.
- Update `SKILL.md` so the skill explicitly asks for host-native structured
  confirmation when `current_confirmation` exists.
- Add `portable_question` to `current_confirmation`.

### Phase 2

- Formalize the confirmation artifact schema under `contract/`.
- Extend shared docs so other configuration skills can reuse this contract.

### Phase 3

- Extend the shared `skill.yaml` schema so skills can declare structured
  confirmation metadata in a platform-neutral way.
- Let host runtimes auto-render `portable_question` without relying only on
  prompt adherence.

## Future Extensions

- richer option annotations for previews and explanations
- explicit normalization rules for mapping free-text replies back to
  `field=value`
- host capability negotiation beyond the current common 2-4 option limit
