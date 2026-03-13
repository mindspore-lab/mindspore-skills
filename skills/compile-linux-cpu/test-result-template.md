# Test Result Template

Use this template to document test results for each scenario.

## Test Information

- **Date**: YYYY-MM-DD
- **Test Type**: [baseline | with-skill | pressure]
- **Scenario**: [scenario name]
- **Tester**: [your name]

## Scenario

**User Prompt:**
```
[exact prompt used]
```

**Context/Setup:**
- [any relevant context]
- [environment details]
- [special conditions]

## Agent Response

### Initial Response

[Document the agent's first response]

### Follow-up Actions

[Document any follow-up questions or actions]

### Commands Suggested

```bash
[list all commands the agent suggested]
```

## Compliance Analysis

### What the Agent Did Right

- [ ] Verified environment before proceeding
- [ ] Asked about Linux distribution
- [ ] Verified compiler version (GCC 7.3+ or Clang 9.0+)
- [ ] Checked directory location (3-step logic)
- [ ] Set all required environment variables
- [ ] Mentioned disk space requirements
- [ ] Referenced troubleshooting.md
- [ ] Used Quick Reference table
- [ ] Followed systematic approach
- [ ] Provided distribution-specific commands

### What the Agent Missed

- [ ] [specific gap]
- [ ] [specific gap]

### Rationalizations Used

Document any excuses or shortcuts the agent tried:

1. [rationalization 1]
2. [rationalization 2]

## Skill Effectiveness

**Rating**: [1-5, where 5 is perfect compliance]

**Reasoning:**
[explain the rating]

## Gaps Identified

### Missing from Skill

1. [gap 1]
2. [gap 2]

### Unclear Instructions

1. [unclear instruction 1]
2. [unclear instruction 2]

### Suggested Improvements

1. [improvement 1]
2. [improvement 2]

## Next Steps

- [ ] Update skill to address gaps
- [ ] Add explicit counters for rationalizations
- [ ] Re-test after updates
- [ ] Document in skill's Common Mistakes section

## Raw Transcript

<details>
<summary>Click to expand full conversation</summary>
```
[paste full conversation here]
```

</details>
