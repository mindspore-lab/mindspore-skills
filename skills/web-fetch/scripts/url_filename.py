#!/usr/bin/env python3
"""
URL 文件名工具

将 URL 转换为可读且跨平台安全的文件名片段。
"""

import os
import re
from urllib.parse import parse_qsl, unquote, urlparse


def _sanitize_token(value):
    """将任意字符串转换为文件名安全片段。"""
    cleaned = unquote(value or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-")


def _build_query_part(query):
    """将 query 转换为可读片段，例如 q-issue-i1cevz-page-2。"""
    if not query:
        return ""

    tokens = ["q"]
    for key, value in parse_qsl(query, keep_blank_values=True):
        key_token = _sanitize_token(key)
        value_token = _sanitize_token(value)
        if key_token:
            tokens.append(key_token)
        if value_token:
            tokens.append(value_token)

    # query 可能只有特殊字符，兜底保留 raw 信息
    if len(tokens) == 1:
        fallback = _sanitize_token(query)
        if fallback:
            tokens.append(fallback)

    return "-".join(tokens)


def build_readable_url_stem(url):
    """
    生成包含域名、路径、查询参数的可读文件名前缀。
    """
    parsed = urlparse(url)
    parts = []

    host = _sanitize_token(parsed.netloc)
    if host:
        parts.append(host)

    path_parts = [
        _sanitize_token(part)
        for part in parsed.path.split("/")
        if _sanitize_token(part)
    ]
    parts.extend(path_parts)

    query_part = _build_query_part(parsed.query)
    if query_part:
        parts.append(query_part)

    if not parts:
        parts.append("page")

    return "-".join(parts)


def generate_output_filename(url, extension, output_dir=None):
    """
    生成输出文件路径。
    """
    stem = build_readable_url_stem(url)
    filename = f"{stem}.{extension}"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    return filename
