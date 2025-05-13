from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from uuid import uuid4
import subprocess
import os
import humanize
import json
from io import BytesIO
import zipfile
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
import base64
from mutagen.id3 import ID3, APIC

def extract_metadata(path: str) -> dict:
    try:
        audio = MP3(path, ID3=EasyID3)
        raw_tags = ID3(path)

        # Album Cover
        apic = next((tag for tag in raw_tags.values() if isinstance(tag, APIC)), None)
        cover_data_b64 = base64.b64encode(apic.data).decode("utf-8") if apic else None
        cover_mime = apic.mime if apic else None

        # EasyID3 tags
        tags = {
            key: val[0] if isinstance(val, list) else val
            for key, val in audio.items()
        }

        # Technische Zusatzinfos
        tags.update({
            "duration": round(audio.info.length),
            "bitrate": audio.info.bitrate,
            "sample_rate": audio.info.sample_rate,
            "cover": cover_data_b64,
            "cover_mime": cover_mime,
        })

        return tags
    except Exception as e:
        return {"error": str(e)}
    
app = FastAPI()
templates = Jinja2Templates(directory="templates")

DOWNLOAD_DIR = "./tmp/spotdl_downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

download_status = {}


def search_spotdl(query: str) -> list[dict]:
    try:
        result = subprocess.run(
            ["spotdl", "search", query, "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        lines = [json.loads(line) for line in result.stdout.strip().splitlines() if line.strip()]
        return lines
    except subprocess.CalledProcessError:
        return []


def run_spotdl(query: str, file_id: str):
    path = os.path.join(DOWNLOAD_DIR, file_id)
    os.makedirs(path, exist_ok=True)
    download_status[file_id] = "downloading"

    command = ["spotdl", "download", query, "--output", path]
    try:
        subprocess.run(command, check=True)
        download_status[file_id] = "done"
    except subprocess.CalledProcessError:
        download_status[file_id] = "error"


def list_downloaded_files(file_id: str):
    path = os.path.join(DOWNLOAD_DIR, file_id)
    files = []
    for root, _, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith(".mp3"):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, os.path.join(DOWNLOAD_DIR, file_id))
                size = os.path.getsize(full_path)
                files.append({
                    "name": fname,
                    "id": file_id,
                    "size": humanize.naturalsize(size),
                    "rel_path": rel_path.replace("\\", "/"),
                })
    return files


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/start")
async def start_download(query: str = Form(...), bg: BackgroundTasks = None):
    file_id = str(uuid4())
    bg.add_task(run_spotdl, query, file_id)
    return RedirectResponse(url=f"/status_page/{file_id}", status_code=303)


@app.get("/status/{file_id}")
async def check_status(file_id: str):
    status = download_status.get(file_id, "unknown")
    return {"file_id": file_id, "status": status}


@app.get("/file/{file_id}/{filename:path}")
async def get_file(file_id: str, filename: str):
    full_path = os.path.join(DOWNLOAD_DIR, file_id, filename)
    if os.path.exists(full_path) and full_path.endswith(".mp3"):
        return FileResponse(
            full_path, filename=os.path.basename(full_path), media_type="audio/mpeg"
        )
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/status_page/{file_id}", response_class=HTMLResponse)
async def status_page(request: Request, file_id: str):
    status = download_status.get(file_id, "unknown")
    files = list_downloaded_files(file_id)
    return templates.TemplateResponse("status.html", {
        "request": request,
        "file_id": file_id,
        "status": status,
        "files": files,
    })


@app.get("/zip/{file_id}")
async def download_zip(file_id: str):
    path = os.path.join(DOWNLOAD_DIR, file_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Download not found")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".mp3"):
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, path)
                    zipf.write(full_path, arcname)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_id}.zip"'},
    )
    

@app.get("/meta/{file_id}/{filename:path}")
async def get_metadata(file_id: str, filename: str):
    full_path = os.path.join(DOWNLOAD_DIR, file_id, filename)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return JSONResponse(content=extract_metadata(full_path))