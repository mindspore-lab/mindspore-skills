# Skill Examples

`examples/` is the example entrypoint for this repository. It does not store
fake outcomes or machine-specific runtime artifacts. It only maintains two
kinds of content:

- skill panoramas: which real scenarios a skill should cover, what is already
  completed, and what is still missing
- runnable demo documents: prerequisites, recommended prompts, expected agent
  actions, expected artifacts, and success criteria

## Current Directories

- `failure-agent/`: minimal repro scripts
- `readiness-agent/`: real readiness workflow examples
- `performance-agent/`: real performance workflow examples

## How To Use It

1. Start with the skill-specific `README.md`.
2. Find a completed demo document.
3. Prepare the real environment and workspace described by that document.
4. Use the prompt from the document to drive the agent directly.

## Notes

- Older quickstart-note content has been folded into the formal demo documents.
- The formal examples in this directory are the only supported reference entry
  points for example workflows.
