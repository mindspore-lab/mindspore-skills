#!/usr/bin/env bash
# find_ops.sh — Enumerate all overloads for an API and check existing kernels.
#
# Usage:
#   bash skills/cpu-plugin-builder/scripts/find_ops.sh <api_name> <mindspore_root> <plugin_kernel_dir>
#
# Example:
#   bash skills/cpu-plugin-builder/scripts/find_ops.sh sub /path/to/mindspore /path/to/op_plugin/ops/kernel
#
# Output:
#   A table showing each *_op.yaml, the derived PrimitiveName, whether a kernel
#   already exists in plugin_kernel_dir, and the kernel filename if found.
#
# Note: Step 1 (mint alias lookup) is informational and non-blocking.
#   For Tensor.* / nn.functional.* APIs that are not in mint/__init__.py,
#   the script will warn and continue to the operator table.

set -euo pipefail

API_NAME="${1:?Usage: $0 <api_name> <mindspore_root> <plugin_kernel_dir>}"
MS_ROOT="${2:?Missing mindspore_root}"
KERNEL_DIR="${3:?Missing plugin_kernel_dir}"

# ── Step 1: Show alias line from mint/__init__.py (informational, non-blocking) ─
MINT_INIT="${MS_ROOT}/python/mindspore/mint/__init__.py"
echo "=== mint/__init__.py alias for '${API_NAME}' (informational) ==="
if [[ ! -f "$MINT_INIT" ]]; then
  echo "WARNING: $MINT_INIT not found — skipping alias lookup."
  echo "  This is expected for Tensor.*, nn.functional.*, or module-level APIs."
else
  alias_line=$(grep -E "\b${API_NAME}\b" "$MINT_INIT" | head -3 || true)
  if [[ -z "$alias_line" ]]; then
    echo "  '${API_NAME}' not found in mint/__init__.py."
    echo "  This may be a Tensor method, nn.functional op, or alias — continue below."
  else
    echo "$alias_line"
  fi
fi
echo ""

# ── Step 2: Locate api_def directory ─────────────────────────────────────────
# Try the canonical path first, then a few common alternates.
API_DEF_DIR=""
for candidate in \
    "${MS_ROOT}/ops/api_def/${API_NAME}" \
    "${MS_ROOT}/ops/api_def/mint_${API_NAME}" \
    "${MS_ROOT}/mindspore/ops/api_def/${API_NAME}"; do
  if [[ -d "$candidate" ]]; then
    API_DEF_DIR="$candidate"
    break
  fi
done

if [[ -z "$API_DEF_DIR" ]]; then
  echo "ERROR: api_def directory not found for '${API_NAME}'." >&2
  echo "  Tried:" >&2
  echo "    ${MS_ROOT}/ops/api_def/${API_NAME}" >&2
  echo "    ${MS_ROOT}/ops/api_def/mint_${API_NAME}" >&2
  echo "    ${MS_ROOT}/mindspore/ops/api_def/${API_NAME}" >&2
  echo "" >&2
  echo "  Fallback: search manually with:" >&2
  echo "    find ${MS_ROOT} -type d -name '${API_NAME}' | grep api_def" >&2
  exit 1
fi
echo "api_def directory: ${API_DEF_DIR}"
echo ""

# ── Step 3: Collect *_op.yaml files ──────────────────────────────────────────
mapfile -t YAML_FILES < <(find "$API_DEF_DIR" -maxdepth 1 -name '*_op.yaml' | sort)
if [[ ${#YAML_FILES[@]} -eq 0 ]]; then
  echo "ERROR: no *_op.yaml files found in $API_DEF_DIR" >&2
  exit 1
fi

# ── Steps 4–6: CamelCase conversion + kernel lookup + table output ────────────
echo "=== Operator Discovery Table ==="
printf "%-40s %-30s %-10s %s\n" "yaml_file" "PrimitiveName" "status" "kernel_file"
printf "%-40s %-30s %-10s %s\n" \
       "----------------------------------------" \
       "------------------------------" \
       "----------" \
       "-----------"

missing_count=0
for yaml_path in "${YAML_FILES[@]}"; do
  yaml_file=$(basename "$yaml_path")

  # Strip _op.yaml suffix, then snake_case → CamelCase
  base="${yaml_file%_op.yaml}"
  prim_name=$(echo "$base" | awk -F_ '{
    result=""
    for(i=1; i<=NF; i++) result = result toupper(substr($i,1,1)) substr($i,2)
    print result
  }')

  # Search plugin_kernel_dir for a file containing the kernel entry point
  kernel_file=""
  status="missing"
  if [[ -d "$KERNEL_DIR" ]]; then
    match=$(grep -rl "extern \"C\" int ${prim_name}" "$KERNEL_DIR" 2>/dev/null | head -1 || true)
    if [[ -n "$match" ]]; then
      kernel_file=$(basename "$match")
      status="exists"
    else
      missing_count=$((missing_count + 1))
    fi
  fi

  printf "%-40s %-30s %-10s %s\n" "$yaml_file" "$prim_name" "$status" "$kernel_file"
done

echo ""
if [[ $missing_count -eq 0 ]]; then
  echo "RESULT: All primitives already have kernel files."
else
  echo "RESULT: ${missing_count} primitive(s) need new kernel files (status=missing)."
fi
