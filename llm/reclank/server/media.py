"""The media parts of a request: their bytes (a data URI, or a file under
--media-dir) and the budget of an image. The engines encode them."""
from __future__ import annotations

import base64
import os


class MediaSource:
    """Read the media parts of the requests of a server."""

    def __init__(self, media_dir=None):
        # the directory of local media files that a request may name (None:
        # data URIs only)
        self.media_dir = media_dir

    def bytes_of(self, ref):
        """Return the bytes of a data URI, or of a file under media_dir."""
        ref = str(ref or "")
        if ref.startswith("data:"):
            head, _, body = ref.partition(",")
            return base64.b64decode(body) if ";base64" in head else body.encode()
        if ref.startswith(("http://", "https://")):
            raise ValueError("this server does not fetch media from the network; "
                             "send a data URI")
        if self.media_dir is None:
            raise ValueError("send the media as a data URI (the server has no --media-dir)")
        path = os.path.realpath(ref[7:] if ref.startswith("file://") else ref)
        root = os.path.realpath(self.media_dir)
        if not path.startswith(root + os.sep):
            raise ValueError("the media path is not under --media-dir")
        with open(path, "rb") as f:
            return f.read()

    def image(self, part, budget):
        """Return (bytes, budget) of an image part; detail low/high and
        max_soft_tokens change the default budget."""
        v = part.get("image_url", part.get("image", part.get("url")))
        detail = None
        if isinstance(v, dict):
            detail = v.get("detail")
            v = v.get("url")
        budget = {"low": 70, "high": 1120}.get(detail, budget)
        if part.get("max_soft_tokens") is not None:
            budget = int(part["max_soft_tokens"])
        return self.bytes_of(v), budget

    def video(self, part):
        """Return the bytes of a video part (video_url or video: a data URI or
        a path under media_dir)."""
        v = part.get("video_url", part.get("video", part.get("url")))
        if isinstance(v, dict):
            v = v.get("url")
        return self.bytes_of(v)

    def audio(self, part):
        """Return the bytes of an audio part (input_audio data, or a URI)."""
        v = part.get("input_audio", part.get("audio", part.get("url")))
        if isinstance(v, dict):
            if v.get("data") is not None:
                return base64.b64decode(v["data"])
            v = v.get("url")
        return self.bytes_of(v)
