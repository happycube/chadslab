"""The messages of an OpenAI request: the text of a content, its media
parts, and the token ids of a prompt with the spans of its media. The
server and the engines of the families use them."""
from __future__ import annotations

MEDIA_TYPES = ("image", "image_url", "audio", "input_audio", "video", "video_url")


def content_text(content):
    """Return the text of a message content. Take a string or a part list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("text") is not None:
                    out.append(str(part["text"]))
            else:
                out.append(str(part))
        return "".join(out)
    return str(content)


def has_media(content):
    """Return True when a content part list has an image, audio, or video part."""
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in MEDIA_TYPES for p in content)


def message_parts(content):
    """Return a content part list for render_chat: the text parts (type
    text) and the media parts, in order."""
    out = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") in MEDIA_TYPES:
                out.append(part)
            elif part.get("text") is not None:
                out.append({"type": "text", "text": str(part["text"])})
        else:
            out.append({"type": "text", "text": str(part)})
    return out


class PromptIds(list):
    """The token ids of a prompt, with the soft-token spans of its media
    (media.Span) in .spans."""
    spans = ()


def common_len(a, b):
    """Return the count of the first items that a and b share."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
