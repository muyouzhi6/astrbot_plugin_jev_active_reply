"""Bounded burst lifecycle independent of incoming event garbage collection."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

FRAME = "_jev_burst_frame"


@dataclass
class BurstFrame:
    sender: str
    cid: str
    started: float
    last: float
    texts: list
    parts: list
    ids: list
    event_ref: object
    task_ref: object
    size: int
    closed: bool = False
    committed: bool = False
    sending: int = 0
    takeover: bool = False


class BurstBook:
    def __init__(self):
        self.entries = OrderedDict()

    def prune(self, now):
        for key, frame in list(self.entries.items()):
            if frame.closed or now - frame.last > 30:
                del self.entries[key]

    def get(self, sender, cid, now):
        self.prune(now)
        return self.entries.get((sender, cid))

    def put(self, frame):
        key = (frame.sender, frame.cid)
        self.entries[key] = frame
        self.entries.move_to_end(key)
        while (
            len(self.entries) > 32
            or sum(f.size for f in self.entries.values()) > 262144
        ):
            self.entries.popitem(last=False)


def payload_size(texts, parts):
    # Include inline media references; image-count limits alone do not bound RAM.
    total = sum(len(t.encode()) for t in texts)
    for part in walk(parts):
        for field in ("text", "file", "url", "path", "message_str"):
            value = getattr(part, field, None)
            if isinstance(value, str):
                total += len(value.encode())
        total += 128
    return total


def walk(parts):
    stack, seen = list(parts), set()
    while stack:
        part = stack.pop()
        if id(part) in seen:
            continue
        seen.add(id(part))
        yield part
        stack.extend(getattr(part, "chain", None) or [])


def attachments_available(parts):
    for part in walk(parts):
        if type(part).__name__ not in ("Image", "Record", "Video", "File"):
            continue
        value = getattr(part, "url", None) or getattr(part, "file", None) or ""
        if value.startswith("file://"):
            value = unquote(urlsplit(value).path)
        if value.startswith("/") and not Path(value).is_file():
            return False
    return True


def close(event):
    frame = event.get_extra(FRAME)
    if frame:
        frame.closed = True


def transfer_attachments(old, new, parts):
    """Move cleanup ownership only for referenced event-owned local attachments."""
    tracked = set(getattr(old, "_temporary_local_files", ()))
    if not tracked:
        return
    paths = set()
    for part in parts:
        chain = [part, *(getattr(part, "chain", None) or [])]
        for item in chain:
            for field in ("file", "path", "url"):
                value = getattr(item, field, None)
                if not isinstance(value, str):
                    continue
                if value.startswith("file://"):
                    value = unquote(urlsplit(value).path)
                if value.startswith("/"):
                    paths.add(str(Path(value).resolve(strict=False)))
    for path in tracked:
        if str(Path(path).resolve(strict=False)) in paths:
            new.track_temporary_local_file(path)
            untrack = getattr(old, "untrack_temporary_local_file", None)
            if callable(untrack):
                untrack(path)
            else:
                # AstrBot 4.26.8 tracks these paths in a list, without an untrack API.
                old._temporary_local_files.remove(path)
