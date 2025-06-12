import asyncio
from io import BytesIO
import os
import zipfile
import requests
import uuid
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import unquote, urlparse
from fastapi import (
    Cookie,
    FastAPI,
    Request,
    Form,
    HTTPException,
)
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    HTMLResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rapidfuzz import process, fuzz
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC
import base64


# === Configuration ===
SPOTDL_BASE = os.getenv("SPOTDL_BASE", "http://127.0.0.1:8800")
DOWNLOAD_DIR = Path("./downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)
DEFAULT_COOKIE_EXPIRATION = 60 * 60  # 1 hour
SPOTDL_EXEC = "/root/ytloader/mini-dl-api/venv/bin/spotdl"

# === Local file search index ===
file_index: List[dict] = []
download_status = {}


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


# === FastAPI app and templates ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Build file index on startup
class SearchResult(BaseModel):
    filename: str
    score: float
    duration: Optional[int]
    bitrate: Optional[int]
    sample_rate: Optional[int]
    cover: Optional[str]
    cover_mime: Optional[str]


async def build_file_index_helper():
    file_index.clear()
    for root, _, files in os.walk(DOWNLOAD_DIR):
        for f in files:
            abs_path = os.path.join(root, f)
            rel = os.path.relpath(abs_path, DOWNLOAD_DIR)
            file_index.append({"original": rel, "lower": rel.lower()})


@app.on_event("startup")
async def build_file_index():
    await build_file_index_helper()


def extract_metadata(path: Path) -> dict:
    audio = MP3(path, ID3=EasyID3)
    raw = ID3(path)
    apic = next((t for t in raw.values() if isinstance(t, APIC)), None)

    return {
        "duration": round(audio.info.length),
        "bitrate": audio.info.bitrate,
        "sample_rate": audio.info.sample_rate,
        "cover": base64.b64encode(apic.data).decode() if apic else None,
        "cover_mime": apic.mime if apic else None,
        "filesize": path.stat().st_size,
    }


@app.post("/search", response_model=List[SearchResult])
async def search_files(request: SearchRequest):
    query = request.query.lower()
    lower_index = [entry["lower"] for entry in file_index]

    matches = process.extract(
        query,
        lower_index,
        scorer=fuzz.WRatio,
        limit=request.limit * 2,
    )
    seen: Set[str] = set()
    results: List[SearchResult] = []

    for matched_lower, score, index in matches:
        entry = file_index[index]
        rel_path = entry["original"]
        name = Path(rel_path).name

        if score < 50 or name in seen:
            continue
        seen.add(name.lower())
        meta = extract_metadata(DOWNLOAD_DIR / rel_path)
        results.append(
            SearchResult(
                filename=name,
                score=score,
                duration=meta["duration"],
                bitrate=meta["bitrate"],
                sample_rate=meta["sample_rate"],
                cover=meta["cover"],
                cover_mime=meta["cover_mime"],
            )
        )
        if len(results) >= request.limit:
            break

    if not results:
        raise HTTPException(404, "Keine passenden Dateien gefunden.")
    return results


# === Landing page ===
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, client_id: Optional[str] = Cookie(None)):
    if not client_id:
        client_id = str(uuid.uuid4())
        response = templates.TemplateResponse("index.html", {"request": request})

        response.set_cookie(
            "client_id", client_id, httponly=True, max_age=DEFAULT_COOKIE_EXPIRATION
        )
        return response
    else:
        response = templates.TemplateResponse("index.html", {"request": request})
        response.set_cookie(
            "client_id", client_id, httponly=True, max_age=DEFAULT_COOKIE_EXPIRATION
        )
        return templates.TemplateResponse("index.html", {"request": request})


@app.get("/refresh_cookie")
async def refresh_cookie():
    new_id = str(uuid.uuid4())
    response = RedirectResponse(url="/")
    response.set_cookie("client_id", new_id, httponly=True)
    return response


class SongItem(BaseModel):
    name: str
    artists: List[str]
    album_name: Optional[str]
    duration: int
    url: str
    cover_url: Optional[str]
    popularity: int


@app.get("/results", response_class=HTMLResponse)
async def results(
    request: Request, query: str, client_id: Optional[str] = Cookie(None)
):
    if query.startswith("http"):
        asyncio.create_task(trigger_download(query, client_id))
        response = RedirectResponse(f"/status_page/{client_id}", status_code=303)
        response.set_cookie(
            "client_id", client_id, httponly=True, max_age=DEFAULT_COOKIE_EXPIRATION
        )
        return response
    else:
        resp = requests.get(f"{SPOTDL_BASE}/api/songs/search", params={"query": query})
    resp.raise_for_status()
    raw = resp.json()

    songs = [
        SongItem(
            name=s["name"],
            artists=s.get("artists", []),
            album_name=s.get("album_name"),
            duration=s["duration"],
            url=s["url"],
            cover_url=s.get("cover_url"),
            popularity=s.get("popularity"),
        )
        for s in raw
    ]

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "query": query, "songs": songs},
    )


async def run_spotdl(query: str, file_id: str):
    # Status initialisieren
    download_status[file_id] = {
        "status": "downloading",
        "message": "Starte Download...",
        "log": [],
    }

    print(file_id)

    client_path = DOWNLOAD_DIR / file_id
    client_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        SPOTDL_EXEC,
        "download",
        query,
        "--output",
        "{artists} - {title}.{output-ext}",  # relativ, weil cwd gesetzt wird
        "--format",
        "mp3",
        "--bitrate",
        "128k",
        "--simple-tui",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(client_path),  # <<< Hier ist der wichtige Teil!
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdout
    async for raw in proc.stdout:
        line = raw.decode(errors="ignore").strip()
        print(f"[{file_id}] {line}")

        if " - " in line and ":" in line:
            try:
                title, msg = line.split(":", 1)
                msg = msg.strip()
                download_status[file_id]["message"] = msg
                download_status[file_id]["log"].append(msg)
            except Exception:
                pass
        else:
            download_status[file_id]["log"].append(line)

    rc = await proc.wait()
    status = "done" if rc == 0 else "error"
    download_status[file_id]["status"] = status
    download_status[file_id]["message"] = "Fertig" if status == "done" else "Fehler"
    await build_file_index_helper()


# === Helper ===


def is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except:  # noqa: E722
        return False


@app.post("/start")
async def start(
    request: Request,
    query: str = Form(...),
    client_id: Optional[str] = Cookie(None),
):
    if not client_id:
        client_id = str(uuid.uuid4())

    if is_url(query):
        asyncio.create_task(trigger_download(query, client_id))
        response = RedirectResponse(f"/status_page/{client_id}", status_code=303)
        return response
    else:
        resp = requests.get(f"{SPOTDL_BASE}/api/songs/search", params={"query": query})
        resp.raise_for_status()
        songs = resp.json()
        return templates.TemplateResponse(
            "results.html", {"request": request, "query": query, "songs": songs}
        )


# === Status page (nur Template) ===
@app.get("/status_page/{client_id}", response_class=HTMLResponse)
async def status_page(request: Request, client_id: str):
    return templates.TemplateResponse(
        "status.html", {"request": request, "client_id": client_id}
    )


# === Lokaler Download/Listing (optional) ===
@app.get("/files/{file_id}")
async def list_files(file_id: str):
    base = DOWNLOAD_DIR / file_id
    if not base.is_dir():
        raise HTTPException(404, "Not found")
    return [f.relative_to(base).as_posix() for f in base.rglob("*.mp3")]


@app.get("/file/{file_id}/{filename:path}")
async def get_file(file_id: str, filename: str):
    filename = unquote(filename)
    f = DOWNLOAD_DIR / file_id / filename
    print(f"🔍 Datei-Pfad: {f}")
    if not f.exists():
        # Inhalt des Verzeichnisses loggen zur Fehlersuche
        folder = DOWNLOAD_DIR / file_id
        if folder.exists():
            print("📁 Verzeichnisinhalt:", list(folder.glob("*")))
        raise HTTPException(404, f"Datei nicht gefunden: {filename}")
    return FileResponse(f, filename=f.name, media_type="audio/mpeg")


@app.get("/zip/{file_id}")
async def download_zip(file_id: str):
    base = DOWNLOAD_DIR / file_id
    if not base.is_dir():
        raise HTTPException(404, "Not found")
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in base.rglob("*.mp3"):
            zf.write(f, f.relative_to(base))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_id}.zip"'},
    )


@app.get("/file_local/{filename}")
async def download_local(filename: str):
    for rel in file_index:
        if Path(rel["original"]).name == filename:
            fpath = DOWNLOAD_DIR / rel["original"]
            if fpath.exists():
                return FileResponse(fpath, filename=filename, media_type="audio/mpeg")
    raise HTTPException(404, "Datei nicht gefunden")


@app.get("/api/status/{client_id}")
def get_download_status(client_id: str):
    base = DOWNLOAD_DIR / client_id
    if not base.is_dir():
        raise HTTPException(status_code=404, detail="Client-Verzeichnis nicht gefunden")

    mp3_files = base.rglob("*.mp3")

    file_data = []
    for f in mp3_files:
        meta = extract_metadata(f)
        file_data.append(
            {
                "filename": f.name,
                "duration": meta.get("duration"),
                "bitrate": meta.get("bitrate"),
                "sample_rate": meta.get("sample_rate"),
                "cover": meta.get("cover"),
                "cover_mime": meta.get("cover_mime"),
                "filesize": meta.get("filesize"),
            }
        )

    # Hole aktuellen Fortschritt aus globalem Speicher (z. B. run_spotdl() → download_status)
    status = download_status.get(
        client_id, {"status": "idle", "message": "Keine aktiven Downloads", "log": []}
    )

    return {
        "client_id": client_id,
        "status": status["status"],
        "message": status["message"],
        "log": status.get("log", []),
        "files": file_data,
    }


async def run_spotdl_cli(url: str, client_id: str):
    if client_id not in download_status:
        download_status[client_id] = {"status": "pending", "message": "", "log": []}

    download_status[client_id]["status"] = "downloading"
    download_status[client_id]["message"] = "Starte Download..."
    download_status[client_id]["log"].append(f"Starte Download für {url}")

    output_dir = DOWNLOAD_DIR / client_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        SPOTDL_EXEC,
        "download",
        url,
        "--output",
        str("{artists} - {title}.{output-ext}"),
        "--format",
        "mp3",
        "--bitrate",
        "128k",
        "--simple-tui",
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(output_dir),
    )

    download_status[client_id]["status"] = "downloading"
    download_status[client_id]["log"] = []

    async for line in process.stdout:
        decoded = line.decode().strip()
        print(f"[{client_id}] {decoded}")
        download_status[client_id]["message"] = decoded
        download_status[client_id]["log"].append(decoded)

    await process.wait()
    download_status[client_id]["status"] = "done"
    await build_file_index_helper()
    download_status[client_id]["message"] = "Fertig"


async def run_ytdlp(query: str, client_id: str):
    download_status[client_id] = {
        "status": "downloading",
        "message": "Starte YouTube-Download...",
        "log": [],
    }

    client_path = DOWNLOAD_DIR / client_id
    client_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",  # Beste Qualität
        "--embed-thumbnail",
        "--add-metadata",
        "--embed-metadata",
        "--embed-chapters",
        "--prefer-ffmpeg",
        "--yes-playlist",  # Playlists erlaubt
        "--output",
        "%(uploader)s - %(title)s.%(ext)s",  # ← wie gewünscht
        query,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(client_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdout
    async for raw in proc.stdout:
        line = raw.decode(errors="ignore").strip()
        print(f"[yt-dlp:{client_id}] {line}")
        download_status[client_id]["message"] = line
        download_status[client_id]["log"].append(line)

    rc = await proc.wait()
    status = "done" if rc == 0 else "error"
    download_status[client_id]["status"] = status
    download_status[client_id]["message"] = "Fertig" if status == "done" else "Fehler"


def _choose(query: str, prefer: str | None = None):
    if query.startswith(("https://open.spotify.", "spotify:")):
        return "spotify"
    if "youtube.com" in query or "youtu.be" in query:
        return "youtube"
    return prefer or "spotify"


async def trigger_download(url: str, client_id: str) -> None:
    choice = _choose(url)

    if choice == "youtube":
        await run_ytdlp(url, client_id)
    else:
        await run_spotdl_cli(url, client_id)


@app.post("/start_bulk")
async def start_bulk(
    song_urls: List[str] = Form(...),
    client_id: Optional[str] = Cookie(None),
):
    if not client_id:
        client_id = str(uuid.uuid4())

    for url in song_urls:
        asyncio.create_task(trigger_download(url, client_id))

    response = RedirectResponse(f"/status_page/{client_id}", status_code=303)
    response.set_cookie(
        "client_id", client_id, httponly=True, max_age=DEFAULT_COOKIE_EXPIRATION
    )
    return response
