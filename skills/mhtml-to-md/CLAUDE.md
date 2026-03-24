# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python skill that converts MHTML (MIME HTML) web archive files to Markdown format. It extracts content from MHTML files, converts HTML to clean Markdown, and uses Ollama with the glm-ocr:bf16 model for OCR-based text extraction from images.

## Architecture

### Core Components

- **mhtml_to_md.py**: Main conversion script with three key classes:
  - `OllamaExtractor`: Handles OCR requests to Ollama server for image text extraction
  - `MHTMLParser`: Parses MHTML files (MIME multipart format) and extracts HTML/images
  - `HTMLToMarkdownConverter`: Converts HTML content to Markdown using BeautifulSoup

### Content Extraction Strategy

The script uses a multi-tier extraction approach:
1. **Primary**: Looks for GitCode issue drawer structure (`ge-drawer-layer`)
2. **Fallback**: For error pages or non-standard layouts:
   - Extracts page title
   - Detects and extracts error messages
   - Extracts recommended projects with metadata (stars, forks, language)
   - Falls back to main content area

### Image Processing

- Images are extracted from MHTML and processed via Ollama OCR
- GIF and animated images are skipped
- Images smaller than 80x50 pixels are skipped
- HTML tables in OCR results are automatically converted to Markdown tables
- Failed OCR conversions are logged to `ocr_failed.log`

## Development Commands

### Setup

```bash
# Install Python dependencies
pip install pillow requests beautifulsoup4 numpy

# Pull Ollama model (required for OCR)
ollama pull glm-ocr:bf16

# Ensure Ollama server is running
ollama serve
```

### Running the Converter

```bash
# Basic usage (current directory)
python3 mhtml_to_md.py

# Specify input/output directories
python3 mhtml_to_md.py --input /path/to/mhtml --output /path/to/output

# Parallel processing (4 jobs)
python3 mhtml_to_md.py -j 4

# Disable OCR for faster processing
python3 mhtml_to_md.py --no-ocr

# Force re-conversion of all files
python3 mhtml_to_md.py --force
```

### Testing

The `debug/` directory contains test MHTML files for development and validation.

```bash
# Test conversion with debug files
python3 mhtml_to_md.py --input debug --output debug_output

# Test with OCR disabled
python3 mhtml_to_md.py --input debug --output debug_output --no-ocr
```

## Key Implementation Details

### MHTML Parsing

MHTML files are MIME multipart messages. The parser:
- Decodes MIME headers (handles RFC 2047 encoded headers)
- Extracts HTML content from the first text/html part
- Extracts images from image/* parts with base64 decoding
- Handles SSL certificate verification issues

### OCR Integration

- Ollama server must be running on localhost:11434 (configurable)
- OCR prompt is in Chinese for optimal glm-ocr performance
- Supports both English and Chinese text extraction
- Preserves indentation and formatting in code screenshots

### Parallel Processing

- Uses ThreadPoolExecutor for concurrent file processing
- Each thread gets its own Ollama extractor instance
- Ensure Ollama can handle concurrent requests when using `-j > 1`

## Important Notes

- The script requires Ollama to be running for OCR functionality
- Output filenames match input filenames (e.g., `page.mhtml` -> `page.md`)
- Failed conversions are logged but don't stop processing of other files
- The `--force` flag bypasses the skip-if-exists check for re-processing files
