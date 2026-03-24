#!/bin/bash
# Verification script for ascend-ops-reviewer skill

echo "=== Ascend Ops Reviewer Setup Verification ==="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }
echo "✅ Python 3 available"
echo ""

# Check scripts exist
echo "2. Checking scripts..."
for script in fetch_pr.py parse_diff.py analyze_changes.py; do
    if [ -f "scripts/$script" ]; then
        echo "✅ scripts/$script exists"
    else
        echo "❌ scripts/$script missing"
        exit 1
    fi
done
echo ""

# Check reference docs exist
echo "3. Checking reference documents..."
for doc in review_checklist.md common_pitfalls.md torch_npu_patterns.md mindspore_patterns.md; do
    if [ -f "references/$doc" ]; then
        echo "✅ references/$doc exists"
    else
        echo "❌ references/$doc missing"
        exit 1
    fi
done
echo ""

# Check SKILL.md format
echo "4. Checking SKILL.md format..."
if head -1 SKILL.md | grep -q "^---$"; then
    echo "✅ SKILL.md has valid YAML frontmatter"
else
    echo "❌ SKILL.md missing YAML frontmatter"
    exit 1
fi
echo ""

# Test parse_diff.py
echo "5. Testing parse_diff.py..."
if python3 scripts/parse_diff.py 2>&1 | grep -q "Usage:"; then
    echo "✅ parse_diff.py runs correctly"
else
    echo "❌ parse_diff.py has errors"
    exit 1
fi
echo ""

# Test fetch_pr.py
echo "6. Testing fetch_pr.py..."
if python3 scripts/fetch_pr.py 2>&1 | grep -q "Usage:"; then
    echo "✅ fetch_pr.py runs correctly"
else
    echo "❌ fetch_pr.py has errors"
    exit 1
fi
echo ""

# Test analyze_changes.py
echo "7. Testing analyze_changes.py..."
if python3 scripts/analyze_changes.py 2>&1 | grep -q "Usage:"; then
    echo "✅ analyze_changes.py runs correctly"
else
    echo "❌ analyze_changes.py has errors"
    exit 1
fi
echo ""

echo "=== All checks passed! ✅ ==="
echo ""
echo "The ascend-ops-reviewer skill is ready to use."
echo ""
echo "Usage examples:"
echo "  - Review local diff: '检视这个 diff 文件 /path/to/file.diff'"
echo "  - Review PR: 'review 这个 PR https://gitcode.com/mindspore/mindspore/pulls/12345'"
