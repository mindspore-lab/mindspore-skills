#!/bin/bash
# Batch fetch example
# Fetches multiple webpages from a URL list

set -e

# Configuration
URLS_FILE="urls.txt"
OUTPUT_DIR="./output"
FORMAT="mhtml pdf"
CONCURRENCY=2

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if URL list exists
if [ ! -f "$URLS_FILE" ]; then
  echo "Error: $URLS_FILE not found"
  echo "Create a file with one URL per line:"
  echo "  https://example.com/page1"
  echo "  https://example.com/page2"
  exit 1
fi

# Fetch all URLs
echo "Batch fetching from $URLS_FILE..."
python3 scripts/web-fetch-batch.py \
  --urls "$URLS_FILE" \
  --output "$OUTPUT_DIR" \
  --format $FORMAT \
  --concurrency "$CONCURRENCY" \
  --wait 10 \
  --max-wait 30

echo "Done! Check $OUTPUT_DIR for output files."
