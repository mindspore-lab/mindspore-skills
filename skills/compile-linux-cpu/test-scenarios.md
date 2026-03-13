# Test Scenarios for compile-linux-cpu Skill

## Overview

This document contains test scenarios following TDD methodology from writing-skills. Each test should be run with a subagent to verify the skill's effectiveness.

## Test Type

This is a **Technique Skill** - it provides a how-to guide for compilation. Testing focuses on:
- Application scenarios: Can agents apply the technique correctly?
- Variation scenarios: Do they handle edge cases?
- Missing information tests: Do instructions have gaps?

## RED Phase: Baseline Testing (Without Skill)

### Baseline Test 1: Basic Compilation Request

**Scenario:**
```
User: "I need to compile MindSpore from source on my Ubuntu 22.04 machine"
```

**Expected baseline behavior (without skill):**
- Agent may skip environment verification
- May not check if already in source directory
- May forget to set critical environment variables
- May not mention disk space requirements upfront
- May not ask about Linux distribution
- May not verify compiler version
- May not reference troubleshooting resources

**Run with:**
```bash
# Create subagent without skill access
claude --no-skills "I need to compile MindSpore from source on my Ubuntu 22.04 machine. Walk me through the process."
```

**Document:**
- What steps did they skip?
- What verifications did they omit?
- Did they set all required environment variables?
- Did they mention troubleshooting resources?
- Did they ask about Linux distribution?
- Did they verify compiler version?

### Baseline Test 2: Compilation with Existing Directory

**Scenario:**
```
User: "I have a mindspore directory already. How do I compile it on Linux?"
```

**Expected baseline behavior:**
- May not check if it's actually the source directory
- May not ask about updating source code
- May assume directory structure without verification
- May not verify system dependencies are installed

**Run with:**
```bash
claude --no-skills "I have a mindspore directory in my current folder. How do I compile MindSpore from it on Linux?"
```

### Baseline Test 3: Build Failure Scenario

**Scenario:**
```
User: "My MindSpore build on Linux failed with a compiler error. What should I do?"
```

**Expected baseline behavior:**
- May suggest generic solutions without checking troubleshooting.md
- May not verify environment variables are set
- May not check disk space
- May not verify directory location
- May not check compiler version

**Run with:**
```bash
claude --no-skills "I'm trying to compile MindSpore on Ubuntu and got a C++17 error. What should I check?"
```

## GREEN Phase: Testing With Skill

Run the same scenarios WITH the skill loaded and verify:

### Test 1: Basic Compilation (With Skill)

**Expected behavior:**
- Checks conda environment first
- Asks about Linux distribution (Ubuntu/Debian vs CentOS/RHEL)
- Verifies compiler version (GCC 7.3+ or Clang 9.0+)
- Installs system dependencies with correct package manager
- Verifies source directory location (3-step logic)
- Sets all environment variables (CC, CXX, MSLIBS_CACHE_PATH)
- Mentions disk space requirements
- References troubleshooting.md for errors
- Uses Quick Reference table for overview

**Success criteria:**
- All environment variables set before build
- Directory verification follows 3-step logic
- Correct package manager commands for user's distro
- Compiler version verified
- Mentions troubleshooting.md proactively

### Test 2: Existing Directory (With Skill)

**Expected behavior:**
- Checks for build.sh to verify it's source directory
- Asks user about updating source code
- Follows the directory detection logic from Step 4
- Verifies system dependencies are installed

**Success criteria:**
- Verifies directory before proceeding
- Asks about git pull
- Doesn't assume directory structure
- Checks system dependencies

### Test 3: Build Failure (With Skill)

**Expected behavior:**
- First consults reference/troubleshooting.md
- Checks Common Mistakes table
- Verifies environment variables
- Checks disk space
- Verifies compiler version
- Provides context-specific solutions

**Success criteria:**
- References troubleshooting.md first
- Uses Common Mistakes table
- Systematic debugging approach
- Checks compiler version

## REFACTOR Phase: Pressure Testing

### Pressure Test 1: Time Pressure + Missing Dependencies

**Scenario:**
```
User: "I need to compile MindSpore quickly for a demo in 2 hours on Linux. Just give me the commands."
```

**Pressure types:**
- Time pressure (demo deadline)
- User requesting shortcuts ("just the commands")

**Expected rationalizations to watch for:**
- "Let me give you quick commands without explanations"
- "We can skip verification steps to save time"
- "You probably have dependencies installed"
- "Skip asking about Linux distribution"

**Skill should enforce:**
- Verify environment even under time pressure
- Mention 30-60 minute build time upfront
- Check prerequisites before starting
- Ask about Linux distribution
- Verify compiler version
- Reference Quick Reference table for fast overview

### Pressure Test 2: Sunk Cost + Build Failure

**Scenario:**
```
User: "I've been trying to compile on Linux for 3 hours and it keeps failing. I just want to get this working. Can you help?"
```

**Pressure types:**
- Sunk cost (3 hours invested)
- Frustration/exhaustion
- Desire for quick fix

**Expected rationalizations:**
- "Let's try a different approach without checking basics"
- "Skip the troubleshooting doc, let me suggest fixes"
- "Try rebuilding without clearing cache"
- "Don't bother checking compiler version"

**Skill should enforce:**
- Check troubleshooting.md FIRST
- Verify Common Mistakes table
- Systematic approach despite frustration
- Don't skip verification steps
- Check compiler version

### Pressure Test 3: Authority + Non-Standard Setup

**Scenario:**
```
User: "I'm an experienced Linux developer. I have GCC 6.5 and want to use that. How do I compile MindSpore?"
```

**Pressure types:**
- Authority (experienced developer)
- Non-standard request (GCC 6.5 < 7.3)

**Expected rationalizations:**
- "Since you're experienced, you can handle GCC 6.5"
- "Let's try it with 6.5 and see what happens"
- "The 7.3 requirement might be outdated"
- "You can probably work around C++17 issues"

**Skill should enforce:**
- State GCC 7.3+ requirement clearly
- Reference Prerequisites section
- Don't deviate from proven workflow
- Explain why GCC 7.3+ is required (C++17 support)

### Pressure Test 4: Multiple Distributions

**Scenario:**
```
User: "I need to compile on both Ubuntu and CentOS. Give me commands for both."
```

**Pressure types:**
- Multiple targets
- Potential for confusion

**Expected rationalizations:**
- "Let me give you generic commands that work on both"
- "Just use conda for everything, skip system packages"
- "The commands are similar enough"

**Skill should enforce:**
- Provide distribution-specific commands
- Clearly separate Ubuntu/Debian vs CentOS/RHEL instructions
- Don't mix package managers (apt vs yum)
- Verify which distribution user is currently on

## Test Execution Checklist

For each test:
- [ ] Run baseline (without skill) - document exact behavior
- [ ] Run with skill - verify compliance
- [ ] Document any new rationalizations found
- [ ] Update skill if gaps discovered
- [ ] Re-test after skill updates

## Success Criteria

The skill passes testing when agents:
1. Always verify environment before proceeding
2. Ask about Linux distribution for correct package manager
3. Verify compiler version meets requirements (GCC 7.3+ or Clang 9.0+)
4. Follow directory detection logic correctly
5. Set all required environment variables
6. Reference troubleshooting.md for errors
7. Use Common Mistakes table for debugging
8. Maintain systematic approach under pressure
9. Don't skip verification steps even when user requests shortcuts
10. Provide distribution-specific commands

## Running Tests

### Using Claude Code Subagents

```bash
# Baseline test (without skill)
claude --no-skills --prompt "I need to compile MindSpore from source on my Ubuntu 22.04 machine"

# Test with skill
claude --skills compile-linux-cpu --prompt "I need to compile MindSpore from source on my Ubuntu 22.04 machine"
```

### Using Agent Tool

From within Claude Code session:
```
Use Agent tool with prompt: "Test compile-linux-cpu skill with scenario: [scenario description]"
```

## Documenting Results

For each test run, document:
1. **Scenario**: What was tested
2. **Behavior**: What the agent did
3. **Compliance**: Did it follow the skill?
4. **Gaps**: What was missing or wrong?
5. **Rationalizations**: What excuses did it make?

Create a results file: `test-results-YYYY-MM-DD.md`
