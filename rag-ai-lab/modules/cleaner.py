import re

# ---------------------------------------------------------------------------
# 1. Strip residual HTML tags, preserve their text content
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text, preserving their text content.
    Safe for any site — strips <tags> but not the words inside them."""
    return _HTML_TAG_RE.sub(" ", text)


# ---------------------------------------------------------------------------
# 2. Remove "Scripts" trailing artifact
# ---------------------------------------------------------------------------
_SCRIPTS_TRAILING_RE = re.compile(r"\bScripts\s*$", re.MULTILINE)


def strip_scripts_artifact(text: str) -> str:
    """Remove trailing standalone 'Scripts' token — a nav artifact from
    sites that have a JavaScript link block at page bottom."""
    return _SCRIPTS_TRAILING_RE.sub("", text).rstrip()


# ---------------------------------------------------------------------------
# 3. Strip pure contact-info lines at the start of body
# ---------------------------------------------------------------------------
_CONTACT_LINE_RE = re.compile(
    r"^[\s\+\d\(\)\-]+"       # phone number characters
    r"[\s\+\d\(\)\-@\.\w]*"   # optional additional phones/email
    r"\s*$",
)


def strip_leading_contact_header(text: str) -> str:
    """Remove lines at the top of content that contain ONLY phone numbers
    and/or email addresses (no other prose). Safe: only fires if the line
    has zero non-contact-info words."""
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
# 4. Remove "Quick Links" navigation sections
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
    return len(text.split())


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------
def clean_body(text: str) -> str:
    """Apply all cleaners in order. Safe for content from any website."""
    text = strip_html_tags(text)
    text = strip_scripts_artifact(text)
    text = strip_leading_contact_header(text)
    text = strip_quick_links_sections(text)
    return text.strip()
