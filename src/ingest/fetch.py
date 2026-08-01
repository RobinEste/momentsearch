"""Acquiring the source video into worker scratch space.

Two sources, one contract — a local temp file the sampler can read:
  * upload  — the browser already PUT the file to object storage via a
              presigned URL; we stream it down from the bucket.
  * youtube — yt-dlp downloads a small <=480p video-only stream (CLIP
              downsizes to 224px anyway; 4K would just waste bandwidth).

The temp file is worker scratch, deleted after the run — durable copies live
only in object storage ("nothing on local").
"""
from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

from .. import storage


def scratch_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "momentsearch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_upload(storage_key: str, video_id: str) -> Path:
    suffix = Path(storage_key).suffix or ".mp4"
    dest = scratch_dir() / f"{video_id}{suffix}"
    return storage.download_to(storage_key, dest)


_MAX_REDIRECTS = 5


def _is_internal(address: str) -> bool:
    """Is this IP inside our own network rather than out on the internet?"""
    import ipaddress

    ip = ipaddress.ip_address(address)
    # An IPv4 address smuggled inside IPv6 notation (::ffff:169.254.169.254,
    # 6to4, NAT64) has to be judged on the address it really means.
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def _guard_url(url: str) -> None:
    """Refuse a URL that resolves to an address inside our own network.

    The document endpoint takes a URL from a user and this worker fetches it,
    which is server-side request forgery by construction: without this check,
    `http://169.254.169.254/...` reaches the cloud metadata service, and
    `http://clip:8001/...` reaches a service that is not supposed to be public.
    Registration cannot do this check — it must not touch the source at all, and
    DNS can change between then and now — so it belongs here, at fetch time.

    This is the cheap first pass, and on its own it is bypassable: it resolves
    the name, and the socket layer resolves it again when connecting. A DNS
    entry that answers differently to the two lookups (rebinding) walks straight
    through. `_GuardedConnection` below closes that by checking the address the
    socket ACTUALLY connected to; this function stays because a clear rejection
    before any connection attempt is a better error than one halfway through.
    """
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"Refusing to fetch a {parts.scheme or 'schemeless'} URL.")
    host = parts.hostname
    if not host:
        raise ValueError("URL has no host.")
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise ValueError(f"Cannot resolve {host}: {exc}") from None
    for info in infos:
        if _is_internal(info[4][0]):
            raise ValueError(f"Refusing to fetch {host}: resolves to {info[4][0]}, "
                             "which is inside our own network.")


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-run the address check on every hop.

    A redirect is a second URL the user controls indirectly, so validating only
    the one they submitted checks the wrong thing. CPython already refuses to
    redirect to anything but http/https/ftp; ftp is dropped here too, since this
    opener has no handler for it and an internal FTP server is exactly the sort
    of thing this guard exists for.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _guard_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _guarded_connection(base):
    """Wrap an http.client connection class so it checks where it LANDED.

    This is what closes DNS rebinding. `_guard_url` resolves the name, and the
    socket resolves it again a moment later; a record with a one-second TTL can
    answer "public" to the first and "127.0.0.1" to the second. Checking the
    peer address after connect() catches that, and it runs before any request
    bytes are written, so a rebound host gets no request at all — not even the
    blind kind whose side effect is the whole point.
    """
    class _Guarded(base):
        def connect(self):
            super().connect()
            peer = self.sock.getpeername()[0]
            if _is_internal(peer):
                self.close()
                raise ValueError(f"Refusing to fetch {self.host}: connected to "
                                 f"{peer}, which is inside our own network.")
    return _Guarded


def _safe_opener() -> urllib.request.OpenerDirector:
    """An opener that speaks http(s) and nothing else.

    urllib's default opener carries FileHandler, FTPHandler and DataHandler,
    and `urlopen("file:///etc/hostname")` really does read that file [verified in
    this image]. None of those belong on a path that fetches user-supplied URLs,
    so the opener is built from the handlers we want instead of by subtraction.
    """
    import http.client

    class _PinnedHTTP(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(_guarded_connection(http.client.HTTPConnection), req)

    class _PinnedHTTPS(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(_guarded_connection(http.client.HTTPSConnection), req,
                                context=self._context)

    opener = urllib.request.OpenerDirector()
    for handler in (_PinnedHTTP(), _PinnedHTTPS(),
                    urllib.request.HTTPErrorProcessor(), _GuardedRedirects()):
        opener.add_handler(handler)
    return opener


def fetch_http(url: str, source_id: str, *, max_mb: int, suffix: str = ".pdf",
               timeout: int = 120) -> Path:
    """Download a document over http(s) into worker scratch.

    Two properties matter more than they look:

    * The destination path is derived from source_id, and an existing complete
      file is reused instead of re-downloaded. That makes a second attempt after
      a crash skip the network entirely, which is what "a completed stage is not
      re-run" means for this stage. The YouTube path gets this for free from
      yt-dlp; here it has to be written down.
    * The download goes to a `.part` file and is renamed only once it finishes.
      Without that, a process killed mid-download leaves a truncated file that
      the reuse check above would happily treat as complete — trading one bug
      for a quieter, worse one.
    """
    dest = scratch_dir() / f"{source_id}{suffix}"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[fetch] {source_id}: reusing {dest.name} ({dest.stat().st_size} bytes)")
        return dest

    _guard_url(url)
    limit = max_mb * 1024 * 1024
    part = dest.with_suffix(dest.suffix + ".part")
    part.unlink(missing_ok=True)
    # urllib's default User-Agent is refused by several publishers, arxiv.org
    # among them.
    request = urllib.request.Request(url, headers={"User-Agent": "momentsearch/1.0"})
    size = 0
    with _safe_opener().open(request, timeout=timeout) as response, \
            part.open("wb") as out:
        while chunk := response.read(1 << 16):
            size += len(chunk)
            if size > limit:
                out.close()
                part.unlink(missing_ok=True)
                raise ValueError(f"Document exceeds the {max_mb}MB limit.")
            out.write(chunk)
    if size == 0:
        part.unlink(missing_ok=True)
        raise ValueError("Downloaded document is empty.")
    part.rename(dest)
    return dest


_cookie_path: str | None = None


def _cookiefile() -> str | None:
    """Resolve cookies from a mounted file (YT_COOKIES_FILE) or a base64 secret
    (YT_COOKIES_B64, written to a temp file once). Same code, local or cloud."""
    global _cookie_path
    from ..config import YT_COOKIES_B64, YT_COOKIES_FILE

    # Prefer a real mounted file; if the path is set but missing (e.g. the
    # local YT_COOKIES_FILE got imported to Fly where ./data isn't mounted),
    # fall through to the base64 secret instead of handing yt-dlp a dead path.
    if YT_COOKIES_FILE and Path(YT_COOKIES_FILE).exists():
        return YT_COOKIES_FILE
    if YT_COOKIES_B64:
        if _cookie_path is None:
            import base64
            p = scratch_dir() / "yt_cookies.txt"
            p.write_bytes(base64.b64decode(YT_COOKIES_B64))
            _cookie_path = str(p)
        return _cookie_path
    return None


def _yt_opts(video_id: str, clients: list[str]) -> dict:
    from ..config import YT_JS_RUNTIMES, YT_PROXY_URL, YT_REMOTE_COMPONENTS

    opts = {
        # We only sample frames — audio and resolution don't matter, smaller is
        # better. Prefer a <=480p video-only stream, but fall back through ANY
        # video-only stream (the tv/android/ios clients mostly return adaptive
        # video-only formats) and finally ANY format at all, so this never
        # errors "Requested format is not available".
        "format": ("bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]/"
                   "best[height<=480][ext=mp4]/best[height<=480]/"
                   "bestvideo[ext=mp4]/bestvideo/best"),
        "outtmpl": str(scratch_dir() / f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    # Only override the player client when asked (empty = yt-dlp's default,
    # which is best once a JS runtime is available). Forcing tv/android is a
    # fallback for when the default fails.
    if clients:
        opts["extractor_args"] = {"youtube": {"player_client": clients}}
    # JS runtime + EJS solver — required by modern yt-dlp to extract YouTube
    # formats at all (see config). Node ships in the Docker image.
    if YT_JS_RUNTIMES:
        opts["js_runtimes"] = {r: {} for r in YT_JS_RUNTIMES}
    if YT_REMOTE_COMPONENTS:
        opts["remote_components"] = list(YT_REMOTE_COMPONENTS)
    # Cookies are the durable fix when the IP itself is blocked (datacenter);
    # a residential proxy is the alternative. See .env.example.
    cookies = _cookiefile()
    if cookies:
        opts["cookiefile"] = cookies
    if YT_PROXY_URL:
        opts["proxy"] = YT_PROXY_URL
    return opts


def _yt_download(url: str, video_id: str, clients: list[str]) -> tuple[Path, str]:
    import yt_dlp

    with yt_dlp.YoutubeDL(_yt_opts(video_id, clients)) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    return path, (info.get("title") or video_id)


def fetch_youtube(url: str, video_id: str) -> tuple[Path, str]:
    """Download via yt-dlp. Returns (path, title).

    Uses a robust multi-client list on the first try (see _yt_opts). If it
    still fails, retries once with the wider fallback set. Persistent failure
    across ALL videos usually means the IP is blocked (datacenter deploy) —
    set YT_COOKIES_FILE or YT_PROXY_URL then (see .env.example)."""
    from ..config import YT_PLAYER_CLIENTS, YT_FALLBACK_CLIENTS

    try:
        return _yt_download(url, video_id, YT_PLAYER_CLIENTS)
    except Exception as exc:
        extra = [c for c in YT_FALLBACK_CLIENTS if c not in YT_PLAYER_CLIENTS]
        if extra:
            print(f"[fetch] {video_id}: {str(exc)[:80]}… — retrying with "
                  f"{YT_PLAYER_CLIENTS + extra}")
            return _yt_download(url, video_id, YT_PLAYER_CLIENTS + extra)
        raise
