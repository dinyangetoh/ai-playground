"""
Content cleaner for RAG pipeline.

`CleanerConfig` controls which rules are active. Only HTML stripping is on
by default; all site-specific rules are opt-in to keep the cleaner portable
across different websites.
"""

import re

from bs4 import BeautifulSoup
from pydantic import BaseModel


class CleanerConfig(BaseModel):
    """Controls which cleaning rules fire. Site-specific rules default to off."""

    strip_html: bool = True
    """Strip residual HTML tags using BeautifulSoup (safe on plain text too)."""

    strip_scripts_artifact: bool = False
    """Remove trailing standalone 'Scripts' token — a nav artifact specific to
    sites with a JavaScript link block at page bottom."""

    strip_contact_header: bool = False
    """Remove lines at the top of content that contain ONLY phone numbers and/or
    email addresses (no other prose). Enabled when crawling sites whose pages
    start with a contact-info header."""

    strip_quick_links: bool = False
    """Remove '##### Quick Links' headings and the nav-label lines that immediately
    follow. Enabled when crawling sites with Quick Links navigation sections."""


# Default config — only generic, safe rules enabled.
DEFAULT_CLEANER_CONFIG = CleanerConfig()


# ---------------------------------------------------------------------------
# 1. Strip residual HTML tags, preserve text content
# ---------------------------------------------------------------------------

def strip_html_tags(html: str) -> str:
    """Remove HTML markup using BeautifulSoup.

    Handles malformed HTML, CDATA, and embedded script content correctly.
    Safe to call on content that has already had tags removed — returns the
    text unchanged.
    """
    return BeautifulSoup(html, "lxml").get_text(separator=" ")


# ---------------------------------------------------------------------------
# 2. Remove "Scripts" trailing artifact  [site-specific, opt-in]
# ---------------------------------------------------------------------------
_SCRIPTS_TRAILING_RE = re.compile(r"\bScripts\s*$", re.MULTILINE)


def strip_scripts_artifact(text: str) -> str:
    """Remove trailing standalone 'Scripts' token."""
    return _SCRIPTS_TRAILING_RE.sub("", text).rstrip()


# ---------------------------------------------------------------------------
# 3. Strip pure contact-info lines at start of body  [site-specific, opt-in]
# ---------------------------------------------------------------------------
_CONTACT_LINE_RE = re.compile(
    r"^[\s\+\d\(\)\-]+"       # phone number characters
    r"[\s\+\d\(\)\-@\.\w]*"   # optional additional phones/email
    r"\s*$",
)


def strip_leading_contact_header(text: str) -> str:
    """Remove lines at the top of content that contain ONLY phone numbers
    and/or email addresses (no other prose)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if _CONTACT_LINE_RE.match(stripped):
            i += 1
        else:
            break
    return "\n".join(lines[i:]).lstrip()


# ---------------------------------------------------------------------------
# 4. Remove "Quick Links" navigation sections  [site-specific, opt-in]
# ---------------------------------------------------------------------------
_QUICK_LINKS_HEADING_RE = re.compile(
    r"^#{1,6}\s+Quick Links\s*$", re.IGNORECASE | re.MULTILINE
)


def strip_quick_links_sections(text: str) -> str:
    """Remove '##### Quick Links' headings and the nav-label lines that
    immediately follow. Only fires when the heading is exactly 'Quick Links'
    (any level) and the trailing content has no sentence-ending punctuation."""
    result = []
    lines = text.splitlines(keepends=True)
    skip = False
    for line in lines:
        if _QUICK_LINKS_HEADING_RE.match(line.rstrip()):
            skip = True
            continue
        if skip:
            if line.startswith("#"):
                skip = False
                result.append(line)
            elif re.search(r"[.!?]", line):
                # Real prose — stop skipping
                skip = False
                result.append(line)
            # Otherwise nav labels — keep skipping
            continue
        result.append(line)
    return "".join(result)


# ---------------------------------------------------------------------------
# 5. Word count helper (for stub-page detection)
# ---------------------------------------------------------------------------
def word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in *text*."""
    return len(text.split())


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------
def clean_body(
    text: str,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
) -> str:
    """Apply cleaners according to *config*.

    Only `strip_html` is on by default. Site-specific rules must be explicitly
    enabled in the config passed by the caller so that the cleaner remains
    portable across different websites.
    """
    if config.strip_html:
        text = strip_html_tags(text)
    if config.strip_scripts_artifact:
        text = strip_scripts_artifact(text)
    if config.strip_contact_header:
        text = strip_leading_contact_header(text)
    if config.strip_quick_links:
        text = strip_quick_links_sections(text)
    return text.strip()
