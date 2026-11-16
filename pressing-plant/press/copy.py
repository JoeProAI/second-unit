"""Release copy: a per-release X/YouTube post template. No hashtags, ever.
"""
from __future__ import annotations

from .ingest import IngestReport

TEMPLATE = """{title} by {artist}

New press from the plant: full stem pack and the official video.

Stems (vocals, drums, bass, instrumental{extra_stems}) are up for
remixes. Grab the pack, cut it up, send back what you make.

Video: {video_line}
Stems: {stems_line}
"""


def release_copy(report: IngestReport,
                 video_url: str = "(link when rendered)",
                 stems_url: str = "(link when uploaded)") -> str:
    extra = [n for n in report.stems
             if n not in {"vocals", "drums", "bass", "instrumental"}]
    text = TEMPLATE.format(
        title=report.meta.title,
        artist=report.meta.artist,
        extra_stems=(", " + ", ".join(extra)) if extra else "",
        video_line=video_url,
        stems_line=stems_url,
    ).strip()
    # house rule: hard fail rather than ship a hashtag
    if "#" in text:
        raise ValueError("release copy contains a hashtag")
    return text + "\n"
