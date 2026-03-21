#!/bin/bash
# Single webpage fetch example
# Fetches a single webpage and saves as MHTML

set -e

# Configuration
URL="https://example.com"
OUTPUT_DIR="./output"
FORMAT="mhtml"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Fetch webpage
echo "Fetching $URL..."
python3 scripts/web-fetch.py "$URL" \
  --output "$OUTPUT_DIR" \
  --format "$FORMAT" \
  --wait 10 \
  --max-wait 30

echo "Done! Check $OUTPUT_DIR for output files."
