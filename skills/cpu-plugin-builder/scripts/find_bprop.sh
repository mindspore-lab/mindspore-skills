#!/usr/bin/env bash
# find_bprop.sh — Locate and enumerate all primitive ops in a bprop body.
#
# Usage:
#   bash skills/cpu-plugin-builder/scripts/find_bprop.sh <PrimitiveName> <mindspore_root>
#
# Example:
#   bash skills/cpu-plugin-builder/scripts/find_bprop.sh SubExt /path/to/mindspore
#
# Output:
#   1. File + line of the REG_BPROP_BUILDER registration
#   2. Raw bprop body (up to 60 lines)
#   3. Extracted primitive list: Emit("X") and ib->X() calls
#   4. BinopGradCommon note if detected

set -euo pipefail

PRIM_NAME="${1:?Usage: $0 <PrimitiveName> <mindspore_root>}"
MS_ROOT="${2:?Missing mindspore_root}"

CCSRC_DIR="${MS_ROOT}/ccsrc"
if [[ ! -d "$CCSRC_DIR" ]]; then
  echo "ERROR: directory not found: $CCSRC_DIR" >&2
  exit 1
fi

if ! command -v rg &>/dev/null; then
  echo "ERROR: 'rg' (ripgrep) is required but not found. Install ripgrep and retry." >&2
  exit 1
fi

# ── Step 1: Find REG_BPROP_BUILDER registration ───────────────────────────────
echo "=== Searching for REG_BPROP_BUILDER(\"${PRIM_NAME}\") ==="
match=$(rg -n "REG_BPROP_BUILDER\(\"${PRIM_NAME}\"" "$CCSRC_DIR" 2>/dev/null | head -1 || true)

if [[ -z "$match" ]]; then
  echo "NOT FOUND: No C++ bprop builder registered for '${PRIM_NAME}'."
  echo ""
  echo "Fallback options:"
  echo "  1. Python-level bprop — search with:"
  echo "       rg -n '${PRIM_NAME}' ${MS_ROOT}/mindspore/ops/_grad/"
  echo "  2. Primitive name may differ — check the *_op.yaml for the exact class name."
  echo "  3. Alias — try adjacent names (e.g. without 'Ext' suffix)."
  exit 0
fi

echo "$match"
bprop_file=$(echo "$match" | cut -d: -f1)
start_line=$(echo "$match" | cut -d: -f2)
echo ""

# ── Step 2: Print bprop body (up to 60 lines) ────────────────────────────────
echo "=== Bprop body (${bprop_file}, line ${start_line}) ==="
body=$(tail -n +"$start_line" "$bprop_file" | head -60)
echo "$body"
echo ""

# ── Step 3: Extract all primitive ops referenced in the body ──────────────────
echo "=== Primitive ops found in bprop body ==="

# Emit("PrimName") calls
emit_prims=$(echo "$body" | grep -oE 'Emit\("[A-Za-z_0-9]+"' | grep -oE '"[A-Za-z_0-9]+"' | tr -d '"' | sort -u || true)

# ib->MethodName( calls — captures IRBuilder method names (ops like Neg, Rsqrt, etc.)
ib_prims=$(echo "$body" | grep -oE 'ib->([A-Z][A-Za-z0-9]+)\(' | grep -oE '[A-Z][A-Za-z0-9]+' | sort -u || true)

all_prims=$(printf '%s\n%s\n' "$emit_prims" "$ib_prims" | grep -v '^$' | sort -u || true)

if [[ -z "$all_prims" ]]; then
  echo "  (none detected automatically — review raw body above manually)"
else
  while IFS= read -r p; do
    echo "  - $p"
  done <<< "$all_prims"
fi
echo ""

# ── Step 4: Named pattern detection ──────────────────────────────────────────
if echo "$body" | grep -q "BinopGradCommon"; then
  echo "PATTERN: BinopGradCommon detected."
  echo "  -> Expands to: SumExt (or ReduceSum) + Reshape"
  echo "  -> Required kernel files: sum_ext.cc and reshape.cc"
  echo "  -> Check op_plugin/ops/kernel/ — if they exist, no action needed."
  echo ""
fi

# Warn if body was likely truncated (no closing brace found in 60 lines)
close_brace=$(echo "$body" | grep -c '^\s*};\s*$' || true)
if [[ "$close_brace" -eq 0 ]]; then
  echo "WARNING: bprop body may be truncated (no closing '};' in first 60 lines)."
  echo "  -> Review the full file manually starting at line ${start_line} in:"
  echo "     ${bprop_file}"
  echo ""
fi

echo "RESULT: $(echo "$all_prims" | grep -c . 2>/dev/null || echo 0) primitive op(s) extracted. Run find_ops.sh for each to check kernel status."
