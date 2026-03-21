---
name: "mhtml-to-md"
description: "Converts MHTML files to Markdown format with OCR support using Ollama (glm-ocr:bf16) model. Invoke when user wants to convert .mhtml files to .md files or extract content from web archives."
---

# MHTML to Markdown Converter

This skill converts MHTML (MIME HTML) files to Markdown format, extracting the main content and converting images to text using OCR.

## Features

- Extracts content from MHTML web archive files
- Converts HTML to clean Markdown format
- Uses Ollama (glm-ocr:bf16) model for text extraction from images
- Automatic HTML table to markdown table conversion
- Preserves original layout and formatting
- Supports batch processing of multiple files
- Supports parallel processing for faster conversion
- Skips already converted files (avoid duplicate processing)
- Fallback content extraction for error pages or non-standard layouts
- Records failed conversions to `ocr_failed.log`

## Requirements

### Ollama Setup

1. Install Ollama: https://ollama.ai
2. Pull the glm-ocr model:
```bash
ollama pull glm-ocr:bf16
```

### Python Dependencies

```bash
pip install pillow requests beautifulsoup4 numpy
```

## Usage

### Basic Usage

```bash
python3 mhtml_to_md.py --input <input_directory> --output <output_directory>
```

### Arguments

- `--input` or `-i`: Input directory containing MHTML files (default: current directory)
- `--output` or `-o`: Output directory for Markdown files (default: same as input directory)
- `--no-ocr`: Disable OCR processing for images
- `--force`: Force re-conversion even if output file exists
- `--jobs` or `-j`: Number of parallel jobs for processing files (default: 1, sequential)

### Examples

Convert all MHTML files in current directory:
```bash
python3 mhtml_to_md.py
```

Convert files with 4 parallel jobs:
```bash
python3 mhtml_to_md.py -j 4
```

Convert files from a specific directory:
```bash
python3 mhtml_to_md.py --input /path/to/mhtml/files --output /path/to/output
```

Disable OCR for faster processing:
```bash
python3 mhtml_to_md.py --no-ocr
```

Force re-conversion of all files:
```bash
python3 mhtml_to_md.py --force
```

## Output

- Each MHTML file is converted to a corresponding `.md` file
- Output filename matches the input filename (e.g., `page.mhtml` -> `page.md`)
- Images are converted to text using OCR
- HTML tables in OCR results are automatically converted to markdown tables
- Failed conversions are logged to `ocr_failed.log` in the output directory

## Content Extraction

The skill uses a multi-tier content extraction strategy:

1. **Primary extraction**: Looks for GitCode issue drawer structure (`ge-drawer-layer`)
2. **Fallback extraction**: For error pages or non-standard layouts:
   - Extracts page title
   - Detects error pages and extracts error messages
   - Extracts recommended projects with formatted output (stars, forks, language)
   - Falls back to main content area extraction

## OCR Engine

The skill uses Ollama with glm-ocr:bf16 as the OCR engine, which provides:
- High accuracy text recognition
- Automatic table detection and markdown table output
- Excellent performance on code screenshots
- Support for both English and Chinese text
- Proper preservation of indentation and formatting

### OCR Prompt

The OCR uses the following prompt for best results:
```
提取这张截图里的文字内容。要求：
1. 严格保留原始缩进和格式
2. 保留所有括号、引号、冒号等符号
3. 输出纯文本，不要markdown格式
4. 不要任何额外解释
```

### Table Handling

When Ollama returns HTML table content (containing `<table>` tags), it is automatically converted to markdown table format.

## Parallel Processing

Use the `-j` option to enable parallel processing of multiple MHTML files:

```bash
# Use 2 parallel jobs
python3 mhtml_to_md.py -j 2

# Use 4 parallel jobs (recommended for multi-core CPUs)
python3 mhtml_to_md.py -j 4
```

**Note**: When using parallel processing with OCR, ensure your Ollama server can handle multiple concurrent requests.

## Error Handling

- If OCR fails for an image, the conversion of that MHTML file is aborted
- The error is logged to `ocr_failed.log` with timestamp and error details
- The script continues to process the next MHTML file

## Limitations

- GIF and animated images are skipped
- Images smaller than 80x50 pixels are skipped
- Requires Ollama server to be running for OCR functionality
