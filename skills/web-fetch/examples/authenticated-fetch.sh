#!/bin/bash
# Authenticated page fetch example
# Fetches a page requiring login authentication

set -e

# Configuration
URL="https://example.com/protected"
OUTPUT_DIR="./output"
FORMAT="mhtml pdf"
LOGIN_TIMEOUT=600

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Fetching authenticated page: $URL"
echo "If login is required, a browser window will open."
echo "Complete the login manually, then the script will continue."
echo ""

# Fetch authenticated page
python3 scripts/web-fetch.py "$URL" \
  --output "$OUTPUT_DIR" \
  --format $FORMAT \
  --login-timeout "$LOGIN_TIMEOUT" \
  --wait 10 \
  --max-wait 30

echo ""
echo "Done! Check $OUTPUT_DIR for output files."
echo "Cookies have been saved for future fetches from this domain."
