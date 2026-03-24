# Readiness-Agent Env-Fix Policy

This reference defines the policy for the native `env-fix` capability inside
`readiness-agent`.

Use it whenever the skill needs to decide:

- whether remediation is allowed
- whether a blocker is in fix scope
- whether confirmation is required
- what actions are forbidden even when the skill is in fix mode

## Core Rule

`env-fix` is a scoped, target-driven remediation capability.

It exists to repair missing pieces that belong to the dependency closure of the
selected execution target.

It does not exist to generically "clean up the machine" or to perform broad
environment mutation.

## Entry Conditions

Enter `env-fix` only when all of the following are true:

- the active execution strategy allows remediation
- the blocker is inside allowed fix scope
- the blocker is classified as remediable
- the remediation action is explicit enough to avoid guesswork
- the revalidation scope is known

If any of these conditions fail:

- do not enter `env-fix`
- keep the issue as unresolved and reflect it in final status

## Fix Scope

The default native fix scope is:

- `safe-user-space`

This scope allows remediation only inside:

- the permitted workspace
- the selected Python environment
- user-level tool resolution such as PATH visibility for approved tools

It does not allow:

- system-layer installation or upgrade
- global machine mutation unrelated to the selected execution target
- speculative repair without explicit target evidence

## Allowed Remediation Actions

The MVP `env-fix` policy allows these action families:

- install `uv`
- repair PATH so `uv` is directly resolvable
- create or reuse a selected Python environment
- install missing runtime Python dependencies
- install or replace framework packages inside the selected environment
- download a model or required asset only when network use is allowed and the
  action is confirmed

These actions must remain target-scoped and closure-justified.

## Forbidden Remediation Actions

The native `env-fix` policy must never allow:

- driver installation or upgrade
- firmware installation or upgrade
- CANN installation or upgrade
- system Python mutation
- destructive workspace deletion
- silent replacement of ambiguous packages
- mutation of model, dataset, checkpoint, or config files
- distributed-launch setup

If a blocker would require any of the above:

- do not treat it as a native `env-fix` action

## Confirmation Policy

Require explicit user confirmation before:

- installing `uv`
- editing shell profiles or PATH
- creating a new Python environment
- replacing an already installed framework package
- downloading a model or asset from the network

Once the user has confirmed the selected Python environment and remediation
intent, the skill may proceed without extra per-package confirmation only for:

- clearly identified missing runtime Python dependencies
- clearly identified missing framework packages
- clearly identified framework replacements already justified by compatibility
  evidence

If the required package name or target version remains ambiguous:

- stop and report the ambiguity instead of guessing

## Relationship To Blocker Categories

Typical native `env-fix` candidates:

- `env_remediable`
- `framework_remediable`
- some `asset_remediable`

Typical non-native `env-fix` candidates:

- `system_fatal`
- most `workspace_manual`
- unresolved `unknown`

Rules:

- `system_fatal` is never handled by native `env-fix`
- `workspace_manual` is not auto-repaired by native `env-fix`
- `unknown` is not auto-repaired unless the ambiguity is resolved first

## Relationship To Dependency Closure

Every remediation action must be justified by dependency closure evidence.

This means:

- fix only what the selected execution target actually needs
- do not install packages just because they are commonly useful
- do not broaden repair scope beyond the selected target

Important principle:

- no closure evidence, no fix action

## Revalidation Rule

Every successful remediation action must declare a revalidation scope and be
followed by targeted revalidation.

Typical examples:

- `uv` install -> rerun tool resolution checks
- environment creation -> rerun Python environment and framework checks
- package install -> rerun import and framework smoke checks
- model download -> rerun asset presence and model-load checks

Never emit final certification immediately after mutation without revalidation.

## Action Quality Rule

A valid native `env-fix` action must be:

- scoped
- explicit
- reversible in reasoning
- justified by evidence
- paired with revalidation

If an action is broad, ambiguous, or weakly justified:

- do not execute it as native `env-fix`

## Status Guidance

If a remediable blocker is fixed and revalidation succeeds:

- the blocker may be cleared
- final status may improve accordingly

If a remediable blocker is fixed but revalidation fails:

- keep or escalate the blocker
- do not emit `READY`

If a blocker is in theory remediable but confirmation is denied:

- leave it unresolved
- reflect that unresolved state in final status

## Invariants

The native `env-fix` policy must preserve these rules:

- no system-layer mutation
- no mutation outside safe user-space scope
- no fix without closure justification
- no fix without known revalidation scope
- no `READY` after mutation unless revalidation succeeds
