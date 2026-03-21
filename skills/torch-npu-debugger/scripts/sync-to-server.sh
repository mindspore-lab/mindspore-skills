#!/bin/bash
# Sync torch-npu-debugger skill to a remote server.
#
# Usage:
#   bash scripts/sync-to-server.sh user@server [remote_dir]
#
# Arguments:
#   user@server  - SSH target (required)
#   remote_dir   - Destination path (default: ~/torch-npu-debugger)
#
# Examples:
#   bash scripts/sync-to-server.sh dev@192.168.1.100
#   bash scripts/sync-to-server.sh dev@ascend-server /home/dev/skills/torch-npu-debugger

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -lt 1 ]; then
    echo "Usage: $0 user@server [remote_dir]"
    echo ""
    echo "Examples:"
    echo "  $0 dev@192.168.1.100"
    echo "  $0 dev@ascend-server /home/dev/skills/torch-npu-debugger"
    exit 1
fi

TARGET="$1"
REMOTE_DIR="${2:-~/torch-npu-debugger}"

echo "Syncing skill to ${TARGET}:${REMOTE_DIR} ..."

rsync -avz --delete \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='*.plan.md' \
    "${PROJECT_DIR}/" \
    "${TARGET}:${REMOTE_DIR}/"

echo ""
echo "Done. Next steps on the remote server:"
echo ""
echo "  1. Install Claude Code (if not already installed):"
echo "     npm install -g @anthropic-ai/claude-code"
echo ""
echo "  2. Register the skill:"
echo "     mkdir -p ~/.claude/skills"
echo "     ln -sf \"${REMOTE_DIR}\" ~/.claude/skills/torch-npu-debugger"
echo ""
echo "  3. Start Claude Code in your torch_npu workspace:"
echo "     cd ~/torch_npu && claude"
