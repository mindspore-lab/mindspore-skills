#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
import io
import ssl
import json
import argparse
import base64
import requests
import urllib3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.header import decode_header
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class OCRExtractionError(Exception):
    pass

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

OLLAMA_AVAILABLE = True


def html_table_to_markdown(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            return None
        
        markdown_tables = []
        for table in tables:
            rows = []
            for tr in table.find_all('tr'):
                cells = []
                for cell in tr.find_all(['th', 'td']):
                    text = cell.get_text(strip=True).replace('\n', ' ')
                    cells.append(text)
                if cells:
                    rows.append(cells)
            
            if rows:
                md_lines = []
                md_lines.append('| ' + ' | '.join(rows[0]) + ' |')
                md_lines.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
                for row in rows[1:]:
                    md_lines.append('| ' + ' | '.join(row) + ' |')
                markdown_tables.append('\n'.join(md_lines))
        
        return '\n\n'.join(markdown_tables) if markdown_tables else None
    except Exception as e:
        print(f"Failed to convert HTML table to markdown: {e}")
        return None


class OllamaExtractor:
    def __init__(self, model='glm-ocr:bf16'):
        self.model = model
        self._cache = {}
        print(f"Using Ollama model: {model}")
    
    def _image_to_base64(self, image):
        if isinstance(image, str):
            if image.startswith('data:image'):
                match = re.match(r'data:image/\w+;base64,(.+)', image)
                if match:
                    return match.group(1)
                return None
            elif image.startswith('http'):
                response = requests.get(image, timeout=30, verify=False)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Optimize: Only convert if not already RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Normalize dimensions for CogViT patch compatibility (14x14 patches)
        PATCH_SIZE = 14
        MAX_DIM = 1344
        w, h = image.size

        # Cap to max resolution first (preserve aspect ratio)
        # IMPORTANT: Avoid exact MAX_DIM boundary due to CogViT internal bug
        if w > MAX_DIM or h > MAX_DIM:
            scale = min(MAX_DIM / w, MAX_DIM / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # If scaled dimension equals MAX_DIM exactly, reduce by one patch to avoid edge case
            if new_w == MAX_DIM:
                new_w = MAX_DIM - PATCH_SIZE
            if new_h == MAX_DIM:
                new_h = MAX_DIM - PATCH_SIZE
            # Optimize: Use BILINEAR for faster resizing (slight quality trade-off)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            w, h = image.size

        # Pad to nearest multiple of patch size with white pixels
        pad_r = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
        pad_b = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
        if pad_r > 0 or pad_b > 0:
            image = ImageOps.expand(image, border=(0, 0, pad_r, pad_b), fill=(255, 255, 255))

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def extract_text_from_image(self, image):
        try:
            if isinstance(image, str):
                if image.startswith('data:image'):
                    match = re.match(r'data:image/\w+;base64,(.+)', image)
                    if match:
                        image_data = base64.b64decode(match.group(1))
                        image = Image.open(io.BytesIO(image_data))
                    else:
                        return None
                elif image.startswith('http'):
                    response = requests.get(image, timeout=30, verify=False)
                    image = Image.open(io.BytesIO(response.content))
                else:
                    image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if getattr(image, 'format', None) == 'GIF':
                print(f"[DEBUG] Skipping GIF image")
                return None
            
            if hasattr(image, 'n_frames') and image.n_frames > 1:
                print(f"[DEBUG] Skipping animated image with {image.n_frames} frames")
                return None
            
            img_width, img_height = image.size
            if img_width < 80 and img_height < 50:
                print(f"[DEBUG] Skipping small image: {img_width}x{img_height}")
                return None
            
            base64_image = self._image_to_base64(image)
            if not base64_image:
                return None

            payload = {
                "model": self.model,
                "prompt": "提取这张截图里的文字内容。要求：1. 严格保留原始缩进和格式 2. 保留所有括号、引号、冒号等符号 3. 输出纯文本，不要markdown格式 4. 不要任何额外解释",
                "images": [base64_image],
                "stream": False
            }

            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json=payload,
                    timeout=60  # Reduce timeout to fail fast on problematic images
                )

                if response.status_code == 200:
                    result = response.json()
                    if 'response' in result:
                        text = result['response']
                        text = re.sub(r'^```\w*\n', '', text)
                        text = re.sub(r'\n```$', '', text)
                        text = text.strip()

                        if '<table>' in text.lower():
                            md_table = html_table_to_markdown(text)
                            if md_table:
                                return md_table

                        return text
                    raise OCRExtractionError(f"OCR failed: empty response from model")
                else:
                    error_text = response.text[:200]
                    # Check if it's a CogViT crash - skip this image instead of crashing the whole process
                    if 'GGML_ASSERT' in error_text or 'runner process no longer running' in error_text:
                        print(f"[WARN] Skipping image due to CogViT incompatibility")
                        return None  # Skip problematic image gracefully
                    raise OCRExtractionError(f"Ollama API error: {response.status_code} - {error_text}")

            except requests.exceptions.Timeout:
                # Timeout likely means CogViT crashed - skip this image
                print(f"[WARN] OCR timeout, skipping image (likely CogViT incompatibility)")
                return None
            
        except requests.exceptions.ConnectionError:
            raise OCRExtractionError("Ollama server not running. Please start Ollama with: ollama serve")
        except OCRExtractionError:
            raise
        except Exception as e:
            raise OCRExtractionError(f"Ollama extraction failed: {e}")
    
    def extract_text_from_url(self, url):
        if url in self._cache:
            return self._cache[url]
        result = self.extract_text_from_image(url)
        self._cache[url] = result
        return result
    
    def extract_text_from_base64(self, base64_data):
        return self.extract_text_from_image(f"data:image/png;base64,{base64_data}")
    
    def extract_text_from_array(self, image_array):
        return self.extract_text_from_image(image_array)


class MHTMLParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.html_content = None
        self.title = None
        self.resources = {}
        self.images = {}
        self._parse()

    def _decode_mime_header(self, header_value):
        if not header_value:
            return None
        
        decoded_parts = decode_header(header_value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                if charset:
                    try:
                        result.append(part.decode(charset))
                    except:
                        result.append(part.decode('utf-8', errors='replace'))
                else:
                    result.append(part.decode('utf-8', errors='replace'))
            else:
                result.append(part)
        
        return ''.join(result)

    def _parse(self):
        with open(self.filepath, 'rb') as f:
            raw_content = f.read()
        
        subject_match = re.search(rb'Subject: (.+?)(?:\r?\n(?![ \t])|\r?\n$)', raw_content, re.DOTALL)
        if subject_match:
            subject_raw = subject_match.group(1)
            subject_raw = re.sub(rb'\r?\n[ \t]', b'', subject_raw)
            try:
                subject_str = subject_raw.decode('utf-8', errors='replace')
                self.title = self._decode_mime_header(subject_str)
            except:
                pass
        
        boundary_match = re.search(rb'boundary="([^"]+)"', raw_content)
        if not boundary_match:
            boundary_match = re.search(rb'boundary=([^\r\n]+)', raw_content)
        
        if boundary_match:
            boundary = boundary_match.group(1).strip()
            if isinstance(boundary, bytes):
                boundary_str = boundary.decode('utf-8', errors='replace')
            else:
                boundary_str = boundary
            
            boundary_str = boundary_str.lstrip('-')
            
            parts = raw_content.split(b'------' + boundary_str.encode('utf-8'))
        else:
            parts = [raw_content]
        
        for part in parts:
            if b'Content-Type: text/html' in part:
                html_start = part.find(b'<!DOCTYPE') if b'<!DOCTYPE' in part else part.find(b'<html')
                if html_start == -1:
                    html_start = part.find(b'\r\n\r\n')
                    if html_start != -1:
                        html_start += 4
                
                if html_start != -1:
                    raw_html = part[html_start:]
                    raw_html = raw_html.split(b'------MultipartBoundary')[0]
                    
                    self.html_content = self._decode_quoted_printable_bytes(raw_html)
                    break
        
        for part in parts:
            if b'Content-Type: image/' in part:
                content_id_match = re.search(rb'Content-ID: <([^>]+)>', part)
                content_location_match = re.search(rb'Content-Location: ([^\r\n]+)', part)
                encoding_match = re.search(rb'Content-Transfer-Encoding: ([^\r\n]+)', part)
                
                img_start = part.find(b'\r\n\r\n')
                if img_start != -1:
                    img_data = part[img_start + 4:]
                    img_data = img_data.split(b'------MultipartBoundary')[0]
                    
                    encoding = encoding_match.group(1).decode('utf-8').strip().lower() if encoding_match else None
                    
                    if encoding == 'base64':
                        try:
                            img_data = base64.b64decode(img_data)
                        except Exception as e:
                            print(f"Failed to decode base64 image: {e}")
                    elif encoding == 'quoted-printable':
                        img_data = self._decode_quoted_printable_bytes(img_data)
                    
                    content_id = content_id_match.group(1).decode('utf-8') if content_id_match else None
                    content_location = content_location_match.group(1).decode('utf-8') if content_location_match else None
                    
                    if content_id:
                        self.images[f'cid:{content_id}'] = img_data
                    if content_location:
                        self.images[content_location] = img_data

    def _decode_quoted_printable_bytes(self, data):
        data = re.sub(rb'=\r?\n', b'', data)
        
        result = bytearray()
        i = 0
        while i < len(data):
            if data[i:i+1] == b'=' and i + 2 < len(data):
                hex_chars = data[i+1:i+3]
                if re.match(rb'[0-9A-Fa-f]{2}', hex_chars):
                    result.append(int(hex_chars, 16))
                    i += 3
                else:
                    result.append(data[i])
                    i += 1
            else:
                result.append(data[i])
                i += 1
        
        try:
            return result.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Decode error: {e}")
            return result.decode('utf-8', errors='replace')

    def get_html(self):
        return self.html_content
    
    def get_title(self):
        return self.title
    
    def get_images(self):
        return self.images


class HTMLToMarkdownConverter:
    def __init__(self, html_content, title=None, images=None, ocr_extractor=None):
        self.html_content = html_content
        self.title = title
        self.images = images or {}
        self.ocr_extractor = ocr_extractor
        self.ocr_cache = {}

    def _clean_text(self, text):
        text = re.sub(r'\r\n', '\n', text)
        return text

    def _get_drawer_content(self):
        drawer_layer = self.soup.find('div', class_='ge-drawer-layer')
        if drawer_layer:
            drawer = drawer_layer.find('div', class_='ge-drawer')
            if drawer:
                return drawer
        return None

    def _extract_issue_header(self, drawer):
        result = []
        
        if not drawer:
            return result
        
        issue_detail = drawer.find('div', class_='ge-issue-detail')
        if not issue_detail:
            return result
        
        detail_head = issue_detail.find('div', class_='ge-issue-detail-head')
        if detail_head:
            type_icon = detail_head.find('img', class_='ge-issue-type-icon')
            if type_icon:
                issue_type = type_icon.get('title', type_icon.get('alt', ''))
                if issue_type:
                    result.append(f"**Type Icon**: {issue_type}")
            
            issue_id_elem = detail_head.find(string=re.compile(r'#I[A-Z0-9]+'))
            if issue_id_elem:
                issue_id = issue_id_elem.strip()
                result.append(f"**Issue ID**: {issue_id}")
            else:
                copy_btn = detail_head.find('div', class_='ge-copy-button__inner')
                if copy_btn and '#' in copy_btn.get_text():
                    result.append(f"**Issue ID**: {self._clean_text(copy_btn.get_text())}")
            
            state_label = detail_head.find('div', class_='issue-state-label')
            if state_label:
                state_span = state_label.find('span')
                if state_span:
                    result.append(f"**Status**: {self._clean_text(state_span.get_text())}")
        
        return result

    def _extract_issue_title(self, drawer):
        result = []
        
        if not drawer:
            return result
        
        title_div = drawer.find('div', class_='issue-detail__title')
        if title_div:
            title_span = title_div.find('span', class_='fs-24')
            if title_span:
                title_text = self._clean_text(title_span.get_text())
                result.append(f"# {title_text}")
        
        misc_div = drawer.find('div', class_='ge-issue-detail-misc')
        if misc_div:
            items = misc_div.find_all('div', class_='item')
            for item in items:
                text = self._clean_text(item.get_text())
                if text:
                    result.append(f"- {text}")
        
        return result

    def _extract_issue_fields(self, drawer):
        result = []
        
        if not drawer:
            return result
        
        fields_container = drawer.find('div', class_='issue-detail_fields')
        if not fields_container:
            return result
        
        fields = fields_container.find_all('div', class_='issue-detail__field')
        for field in fields:
            label = field.find('div', class_='issue-detail__label')
            if label:
                label_text = self._clean_text(label.get_text())
                
                assignees = field.find_all('div', class_='issue-member-select-selected-member')
                if assignees:
                    names = []
                    for a in assignees:
                        name_span = a.find('span', class_='issue-member-select-selected-member-name')
                        if name_span:
                            names.append(self._clean_text(name_span.get_text()))
                        else:
                            avatar = a.find('span', class_='avatar-img')
                            if avatar:
                                fake = avatar.find('span', class_='fake')
                                if fake:
                                    names.append(self._clean_text(fake.get_text()))
                    if names:
                        result.append(f"**{label_text}**: {', '.join(names)}")
                    continue
                
                selection = field.find('span', class_='ge-select-selection-item')
                if selection:
                    value_text = self._clean_text(selection.get_text())
                    result.append(f"**{label_text}**: {value_text}")
        
        schedule_field = drawer.find('div', class_='ge-issue-detail-schedule-field')
        if schedule_field:
            label = schedule_field.find('div', class_='issue-detail__label')
            start_input = schedule_field.find('input', id=re.compile(r'.*startDate'))
            end_input = schedule_field.find('input', id=re.compile(r'.*endDate'))
            if label and start_input and end_input:
                label_text = self._clean_text(label.get_text())
                start_val = start_input.get('value', '')
                end_val = end_input.get('value', '')
                if start_val or end_val:
                    result.append(f"**{label_text}**: {start_val} ~ {end_val}")
        
        return result

    def _extract_description(self, drawer):
        result = []
        
        if not drawer:
            return result
        
        desc_div = drawer.find('div', class_='ge-issue-detail-description')
        if desc_div:
            markdown_body = desc_div.find('div', class_='markdown-body')
            if markdown_body:
                result.append("\n## Description\n")
                md_content = self._html_to_markdown(markdown_body)
                result.append(md_content)
        
        return result

    def _ocr_image(self, src):
        if src in self.ocr_cache:
            return self.ocr_cache[src]
        
        if not self.ocr_extractor:
            return None
        
        if src.startswith('data:image'):
            match = re.match(r'data:image/\w+;base64,(.+)', src)
            if match:
                base64_data = match.group(1)
                text = self.ocr_extractor.extract_text_from_base64(base64_data)
                self.ocr_cache[src] = text
                return text
        elif src.startswith('cid:'):
            if src in self.images:
                img_data = self.images[src]
                text = self.ocr_extractor.extract_text_from_array(img_data)
                self.ocr_cache[src] = text
                return text
        else:
            if src in self.images:
                img_data = self.images[src]
                text = self.ocr_extractor.extract_text_from_array(img_data)
                self.ocr_cache[src] = text
                return text
            text = self.ocr_extractor.extract_text_from_url(src)
            self.ocr_cache[src] = text
            return text
        
        return None

    def _html_to_markdown(self, element):
        result = []
        
        for child in element.children:
            if child.name is None:
                text = str(child).strip()
                if text:
                    result.append(text)
                continue
            
            if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(child.name[1])
                text = self._clean_text(child.get_text())
                result.append(f"\n{'#' * level} {text}\n")
            elif child.name == 'p':
                text = self._process_inline_elements(child)
                if text:
                    result.append(text)
            elif child.name == 'ul':
                for li in child.find_all('li', recursive=False):
                    text = self._process_inline_elements(li)
                    result.append(f"- {text}")
                result.append("")
            elif child.name == 'ol':
                for i, li in enumerate(child.find_all('li', recursive=False), 1):
                    text = self._process_inline_elements(li)
                    result.append(f"{i}. {text}")
                result.append("")
            elif child.name == 'blockquote':
                text = self._clean_text(child.get_text())
                for line in text.split('\n'):
                    result.append(f"> {line}")
                result.append("")
            elif child.name == 'pre':
                code = child.find('code')
                if code:
                    lang = ''
                    for cls in code.get('class', []):
                        if cls.startswith('language-'):
                            lang = cls[9:]
                            break
                    result.append(f"```{lang}\n{code.get_text()}\n```")
                else:
                    result.append(f"```\n{child.get_text()}\n```")
                result.append("")
            elif child.name == 'code':
                if child.parent and child.parent.name != 'pre':
                    result.append(f"`{child.get_text()}`")
            elif child.name == 'table':
                md_table = self._process_table(child)
                if md_table:
                    result.append('\n' + md_table + '\n')
            elif child.name == 'img':
                alt = child.get('alt', 'Image content')
                src = child.get('src', '')
                
                if self.ocr_extractor and src:
                    ocr_text = self._ocr_image(src)
                    if ocr_text and ocr_text.strip():
                        result.append("\n\n```\n")
                        result.append(ocr_text)
                        result.append("\n```\n")
                    else:
                        result.append(f"\n\n[Image: {alt}]\n")
                else:
                    result.append(f"\n\n[Image: {alt}]\n")
            elif child.name == 'a':
                text = child.get_text()
                href = child.get('href', '')
                result.append(f"[{text}]({href})")
            elif child.name == 'strong' or child.name == 'b':
                text = child.get_text()
                result.append(f"**{text}**")
            elif child.name == 'em' or child.name == 'i':
                text = child.get_text()
                result.append(f"*{text}*")
            elif child.name == 'br':
                result.append("\n")
            elif child.name in ['div', 'span']:
                md = self._html_to_markdown(child)
                if md.strip():
                    result.append(md)
        
        final_result = []
        for item in result:
            if '\n' in item:
                lines = item.split('\n')
                final_result.extend(lines)
            else:
                final_result.append(item)
        
        return '\n'.join(final_result)
    
    def _merge_markdown_tables(self, text):
        lines = text.split('\n')
        result = []
        table_buffer = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            is_table_row = stripped.startswith('|') and stripped.endswith('|')
            is_table_separator = bool(re.match(r'^\|[\s\-:|]+\|$', stripped))
            
            if is_table_row or is_table_separator:
                table_buffer.append(stripped)
                i += 1
                while i < len(lines) and lines[i].strip() == '':
                    i += 1
            else:
                if table_buffer:
                    if len(table_buffer) >= 2:
                        result.append('\n'.join(table_buffer))
                    else:
                        result.extend(table_buffer)
                    table_buffer = []
                result.append(line)
                i += 1
        
        if table_buffer:
            if len(table_buffer) >= 2:
                result.append('\n'.join(table_buffer))
            else:
                result.extend(table_buffer)
        
        return '\n'.join(result)

    def _process_inline_elements(self, element):
        result = []
        for child in element.children:
            if child.name is None:
                result.append(str(child))
            elif child.name == 'strong' or child.name == 'b':
                result.append(f"**{child.get_text()}**")
            elif child.name == 'em' or child.name == 'i':
                result.append(f"*{child.get_text()}*")
            elif child.name == 'code':
                result.append(f"`{child.get_text()}`")
            elif child.name == 'a':
                text = child.get_text()
                href = child.get('href', '')
                result.append(f"[{text}]({href})")
            elif child.name == 'img':
                alt = child.get('alt', 'Image content')
                src = child.get('src', '')
                
                if self.ocr_extractor and src:
                    ocr_text = self._ocr_image(src)
                    if ocr_text and ocr_text.strip():
                        result.append("\n\n```\n")
                        result.append(ocr_text)
                        result.append("\n```\n")
                    else:
                        result.append(f"\n\n[Image: {alt}]\n")
                else:
                    result.append(f"\n\n[Image: {alt}]\n")
            elif child.name == 'br':
                result.append("\n")
            else:
                result.append(child.get_text())
        
        text = ''.join(result)
        text = re.sub(r'\r\n', '\n', text)
        return text

    def _process_table(self, table):
        rows = []
        headers = []
        
        thead = table.find('thead')
        if thead:
            for th in thead.find_all(['th', 'td']):
                headers.append(self._clean_text(th.get_text()))
        
        tbody = table.find('tbody') or table
        for tr in tbody.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cell_text = self._clean_text(td.get_text())
                cells.append(cell_text)
            if cells and cells != headers:
                rows.append(cells)
        
        if not headers and rows:
            headers = rows.pop(0)
        
        if not headers and not rows:
            return ''
        
        result = []
        if headers:
            result.append('| ' + ' | '.join(headers) + ' |')
            result.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        
        for row in rows:
            while len(row) < len(headers):
                row.append('')
            result.append('| ' + ' | '.join(row[:len(headers)]) + ' |')
        
        return '\n'.join(result)

    def _extract_comments(self, drawer):
        result = []
        
        if not drawer:
            return result
        
        comments_container = drawer.find('div', class_='ge-drawer-detail__comments')
        if comments_container:
            result.append("\n## Comments\n")
            
            comment_items = comments_container.find_all('div', class_='ge-comment-item')
            for i, comment in enumerate(comment_items, 1):
                author_elem = comment.find('a', class_='ge-comment-item__author')
                author = self._clean_text(author_elem.get_text()) if author_elem else 'Unknown'
                
                time_elem = comment.find('span', class_='ge-timeago')
                time_str = self._clean_text(time_elem.get_text()) if time_elem else ''
                
                result.append(f"\n### Comment {i} - {author} ({time_str})\n")
                
                content_div = comment.find('div', class_='ge-comment-item__content')
                if content_div:
                    md_content = self._html_to_markdown(content_div)
                    result.append(md_content)
        
        return result

    def _extract_right_fields(self, drawer):
        result = []
        
        if not drawer:
            return result
        
        right_panel = drawer.find('div', class_='ge-issue-detail-content__right')
        if right_panel:
            result.append("\n## Other Fields\n")
            
            field_columns = right_panel.find_all('div', class_='issue-detail__field--column')
            for field in field_columns:
                label = field.find('div', class_='issue-detail__label--column')
                if label:
                    label_text = self._clean_text(label.get_text())
                    
                    selection = field.find('span', class_='ge-select-selection-item')
                    if selection:
                        value_text = self._clean_text(selection.get_text())
                        result.append(f"**{label_text}**: {value_text}")
                        continue
                    
                    labels = field.find_all('span', class_='ge-label')
                    if labels:
                        label_values = [self._clean_text(l.get('title', l.get_text())) for l in labels]
                        result.append(f"**{label_text}**: {', '.join(label_values)}")
                        continue
                    
                    placeholder = field.find('span', class_='ge-select-selection-placeholder')
                    if placeholder:
                        result.append(f"**{label_text}**: {self._clean_text(placeholder.get_text())}")
        
        return result

    def convert(self):
        self.soup = BeautifulSoup(self.html_content, 'html.parser')
        drawer = self._get_drawer_content()
        
        result = []
        
        if drawer:
            result.extend(self._extract_issue_title(drawer))
            result.extend(self._extract_issue_header(drawer))
            result.extend(self._extract_issue_fields(drawer))
            result.extend(self._extract_description(drawer))
            result.extend(self._extract_comments(drawer))
            result.extend(self._extract_right_fields(drawer))
        else:
            print("Warning: Main content not found, extracting basic page content")
            result.extend(self._extract_basic_content())
        
        final_result = '\n'.join(result)
        final_result = re.sub(r'\n{3,}', '\n\n', final_result)
        final_result = self._merge_markdown_tables(final_result)
        
        return final_result.strip()
    
    def _extract_basic_content(self):
        result = []
        
        if self.title:
            result.append(f"# {self.title}\n")
        
        error_page = self.soup.find('div', class_='g-error-page')
        if error_page:
            error_title = error_page.find('div', class_='text-\\[16px\\]')
            if error_title:
                result.append(f"## Error\n\n{error_title.get_text(strip=True)}\n")
            
            back_btn = error_page.find('button')
            if back_btn:
                result.append(f"\n> {back_btn.get_text(strip=True)}\n")
            
            repo_section = error_page.find('div', class_='container_warp')
            if repo_section:
                result.append("\n## Recommended Projects\n")
                repo_cards = repo_section.find_all('div', class_='repo-card-list')
                for card in repo_cards:
                    title_elem = card.find('div', class_='repo-card-list-title')
                    if title_elem:
                        title_span = title_elem.find('span', title=True)
                        if title_span:
                            repo_name = title_span.get('title', title_span.get_text(strip=True))
                            result.append(f"\n### {repo_name}\n")
                    
                    desc_elem = card.find('div', class_='repo-card-list-desc')
                    if desc_elem:
                        desc = desc_elem.get('title', desc_elem.get_text(strip=True))
                        if desc:
                            result.append(f"\n{desc}\n")
                    
                    footer = card.find('div', class_='repo-card-footer')
                    if footer:
                        stats = []
                        star_div = footer.find('div', class_='star-count-one')
                        if star_div:
                            star_text = star_div.get_text(strip=True)
                            if star_text:
                                stats.append(f"⭐ {star_text}")
                        fork_div = footer.find('div', class_='fork-tag')
                        if fork_div:
                            fork_span = fork_div.find('span')
                            if fork_span:
                                stats.append(f"🍴 {fork_span.get_text(strip=True)}")
                        lang_span = footer.find('span', class_='i-language-tag')
                        if lang_span:
                            lang = lang_span.get_text(strip=True)
                            if lang:
                                stats.append(f"💻 {lang}")
                        if stats:
                            result.append(f"\n{' | '.join(stats)}\n")
            
            return result
        
        main_content = self.soup.find('div', class_='devui-layout__content')
        if main_content:
            result.append("\n## Page Content\n")
            original_ocr = self.ocr_extractor
            self.ocr_extractor = None
            try:
                result.append(self._html_to_markdown(main_content))
            finally:
                self.ocr_extractor = original_ocr
            return result
        
        body = self.soup.find('body')
        if body:
            for script in body.find_all('script'):
                script.decompose()
            for style in body.find_all('style'):
                style.decompose()
            result.append("\n## Page Content\n")
            original_ocr = self.ocr_extractor
            self.ocr_extractor = None
            try:
                result.append(self._html_to_markdown(body))
            finally:
                self.ocr_extractor = original_ocr
        
        return result


def get_ocr_extractor(engine='auto'):
    if OLLAMA_AVAILABLE:
        print("Using Ollama (glm-ocr:bf16) as OCR engine")
        return OllamaExtractor()
    else:
        print("No OCR engine available")
        return None


def convert_mhtml_to_markdown(mhtml_path, output_dir=None, ocr_extractor=None, force=False):
    if output_dir is None:
        output_dir = os.path.dirname(mhtml_path)
    
    base_name = os.path.splitext(os.path.basename(mhtml_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.md")
    
    if not force and os.path.exists(output_path):
        print(f"Skipping (already exists): {output_path}")
        return output_path
    
    print(f"Processing: {mhtml_path}")
    
    try:
        parser = MHTMLParser(mhtml_path)
        html_content = parser.get_html()
        title = parser.get_title()
        images = parser.get_images()
        
        if not html_content:
            print(f"Warning: Could not extract HTML content from {mhtml_path}")
            return None
        
        converter = HTMLToMarkdownConverter(html_content, title=title, images=images, ocr_extractor=ocr_extractor)
        markdown_content = converter.convert()
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Generated: {output_path}")
        return output_path
    
    except OCRExtractionError as e:
        error_msg = f"OCR failed for {mhtml_path}: {str(e)}"
        print(f"Error: {error_msg}")
        
        log_path = os.path.join(output_dir if output_dir else os.path.dirname(mhtml_path), "ocr_failed.log")
        with open(log_path, 'a', encoding='utf-8') as f:
            from datetime import datetime
            f.write(f"[{datetime.now().isoformat()}] {error_msg}\n")
        
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Convert MHTML files to Markdown format with OCR support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                              # Convert all .mhtml files in current directory
  %(prog)s -i /path/to/input            # Convert files from specific directory
  %(prog)s -i /path/to/input -o /path/to/output  # Specify input and output directories
  %(prog)s --no-ocr                     # Disable OCR processing for images
  %(prog)s --force                      # Force re-conversion of existing files
        '''
    )
    
    parser.add_argument(
        '-i', '--input',
        default='.',
        help='Input directory containing MHTML files (default: current directory)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory for Markdown files (default: same as input directory)'
    )
    
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR processing for images'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-conversion even if output file exists'
    )
    
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for processing files (default: 1, sequential)'
    )
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output) if args.output else input_dir
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    mhtml_files = glob.glob(os.path.join(input_dir, '*.mhtml'))
    
    if not mhtml_files:
        print(f"No MHTML files found in: {input_dir}")
        sys.exit(0)
    
    print(f"Found {len(mhtml_files)} MHTML file(s)")
    
    if args.no_ocr:
        ocr_extractor = None
        print("OCR disabled by user")
    else:
        ocr_extractor = get_ocr_extractor()
        if ocr_extractor is None:
            print("Warning: No OCR engine available, images will be preserved as text descriptions")
    
    success_count = 0
    fail_count = 0
    
    if args.jobs > 1:
        print(f"Using {args.jobs} parallel jobs")

        # Thread-local storage for OCR extractors (one per thread, not per file)
        thread_local = threading.local()

        def get_thread_ocr_extractor():
            if not hasattr(thread_local, 'ocr_extractor'):
                thread_local.ocr_extractor = None if args.no_ocr else get_ocr_extractor()
            return thread_local.ocr_extractor

        def process_file(mhtml_file):
            try:
                result = convert_mhtml_to_markdown(
                    mhtml_file,
                    output_dir=output_dir,
                    ocr_extractor=get_thread_ocr_extractor(),
                    force=args.force
                )
                return (mhtml_file, result, None)
            except Exception as e:
                import traceback
                return (mhtml_file, None, str(e) + "\n" + traceback.format_exc())

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(process_file, f): f for f in mhtml_files}
            for future in as_completed(futures):
                mhtml_file, result, error = future.result()
                if error:
                    print(f"Error processing {mhtml_file}: {error}")
                    fail_count += 1
                elif result:
                    success_count += 1
                else:
                    fail_count += 1
    else:
        for mhtml_file in mhtml_files:
            try:
                result = convert_mhtml_to_markdown(
                    mhtml_file,
                    output_dir=output_dir,
                    ocr_extractor=ocr_extractor,
                    force=args.force
                )
                if result:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Error processing {mhtml_file}: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1
    
    print(f"\nConversion complete: {success_count} succeeded, {fail_count} failed")


if __name__ == '__main__':
    main()
