#!/bin/bash
# Test runner for compile-linux-cpu skill
# Usage: ./run-tests.sh [baseline|with-skill|pressure]

set -e

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCENARIOS="$SKILL_DIR/test-scenarios.md"
RESULTS_DIR="$SKILL_DIR/test-results"
TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "compile-linux-cpu Skill Test Runner"
echo "========================================="
echo ""

# Function to run a test scenario
run_test() {
    local test_type=$1
    local scenario_name=$2
    local prompt=$3
    local result_file="$RESULTS_DIR/${test_type}-${scenario_name}-${TIMESTAMP}.md"

    echo -e "${YELLOW}Running: ${test_type} - ${scenario_name}${NC}"
    echo ""
    echo "Prompt: $prompt"
    echo ""

    # Create result file header
    cat > "$result_file" <<EOF
# Test Result: ${test_type} - ${scenario_name}
Date: $(date)
Scenario: ${scenario_name}

## Prompt
\`\`\`
$prompt
\`\`\`

## Agent Response
EOF

    # Run the test based on type
    if [ "$test_type" = "baseline" ]; then
        echo "Running WITHOUT skill..."
        echo "(Manual test - observe agent behavior and document in $result_file)"
    elif [ "$test_type" = "with-skill" ]; then
        echo "Running WITH skill..."
        echo "(Manual test - observe agent behavior and document in $result_file)"
    fi

    echo ""
    echo -e "${GREEN}Result file created: $result_file${NC}"
    echo "Please document the agent's behavior in this file."
    echo ""
    echo "---"
    echo ""
}

# Test scenarios
case "${1:-all}" in
    baseline)
        echo "Running BASELINE tests (without skill)..."
        echo ""
        run_test "baseline" "basic-compilation" "I need to compile MindSpore from source on my Ubuntu 22.04 machine. Walk me through the process."
        run_test "baseline" "existing-directory" "I have a mindspore directory in my current folder. How do I compile MindSpore from it on Linux?"
        run_test "baseline" "build-failure" "I'm trying to compile MindSpore on Ubuntu and got a C++17 error. What should I check?"
        ;;

    with-skill)
        echo "Running tests WITH skill..."
        echo ""
        run_test "with-skill" "basic-compilation" "I need to compile MindSpore from source on my Ubuntu 22.04 machine. Walk me through the process."
        run_test "with-skill" "existing-directory" "I have a mindspore directory in my current folder. How do I compile MindSpore from it on Linux?"
        run_test "with-skill" "build-failure" "I'm trying to compile MindSpore on Ubuntu and got a C++17 error. What should I check?"
        ;;

    pressure)
        echo "Running PRESSURE tests..."
        echo ""
        run_test "pressure" "time-pressure" "I need to compile MindSpore quickly for a demo in 2 hours on Linux. Just give me the commands."
        run_test "pressure" "sunk-cost" "I've been trying to compile on Linux for 3 hours and it keeps failing. I just want to get this working. Can you help?"
        run_test "pressure" "authority" "I'm an experienced Linux developer. I have GCC 6.5 and want to use that. How do I compile MindSpore?"
        run_test "pressure" "multiple-distros" "I need to compile on both Ubuntu and CentOS. Give me commands for both."
        ;;

    all)
        echo "Running ALL tests..."
        echo ""
        $0 baseline
        $0 with-skill
        $0 pressure
        ;;

    *)
        echo "Usage: $0 [baseline|with-skill|pressure|all]"
        echo ""
        echo "  baseline    - Run tests without skill (RED phase)"
        echo "  with-skill  - Run tests with skill (GREEN phase)"
        echo "  pressure    - Run pressure tests (REFACTOR phase)"
        echo "  all         - Run all tests"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Test run complete!"
echo "Results directory: $RESULTS_DIR"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Review test results in $RESULTS_DIR"
echo "2. Document agent behavior in each result file"
echo "3. Identify gaps and rationalizations"
echo "4. Update skill if needed"
echo "5. Re-run tests to verify fixes"
