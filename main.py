import os
import time
import shutil
import subprocess
import re
import threading
import collections
import json
import stat
import tempfile
import socket
import urllib.request
import multiprocessing
import asyncio  # ⚡ FIXED: Added missing asyncio import
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any
import psutil
import duckdb
import gradio as gr
import httpx
import requests
from fastapi import FastAPI, HTTPException, Query, Response

# ── Config & Dynamic Hardware Profiling ─────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "aditya7543/Hitek_ImCR_API"
PORT = int(os.environ.get("PORT", "7860"))

# ⚡ FIXED: Pulls token securely from Render Environment Variables first
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# Dynamically scale threads based on host environment (Render vs Local)
SYS_CORES = multiprocessing.cpu_count()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", max(2, SYS_CORES * 2)))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

TEMP_DIR = os.path.join(tempfile.gettempdir(), "duckdb_cache")
os.makedirs(TEMP_DIR, exist_ok=True)
SAFE_TEMP = TEMP_DIR.replace("\\", "/")

# ── 1. BULLETPROOF AUTHENTICATED URLS ───────────────────────────────────────
HF_INDEX_BASE = f"https://__token__:{HF_TOKEN}@huggingface.co/datasets/{REPO_ID}/resolve/main/production_indexes"

# ── 2. L1 In-Memory LRU Cache ───────────────────────────────────────────────
class FastMemoryCache:
    def __init__(self, maxsize: int = 10000, ttl_seconds: int = 7200):
        self.cache: collections.OrderedDict[str, tuple[float, Any]] = collections.OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self.lock:
            if key not in self.cache: return None
            ts, val = self.cache[key]
            if time.time() - ts > self.ttl:
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return val

    def set(self, key: str, val: Any) -> None:
        with self.lock:
            if key in self.cache: self.cache.move_to_end(key)
            self.cache[key] = (time.time(), val)
            if len(self.cache) > self.maxsize: self.cache.popitem(last=False)

MEM_CACHE = FastMemoryCache()

# ── 3. Dynamic Index Discovery (Dual Matrix) ────────────────────────────────
print("🔍 Discovering production shards in /production_indexes on Hugging Face...")
try:
    tree_url = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/production_indexes"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    resp = requests.get(tree_url, headers=headers, timeout=10)
    resp.raise_for_status()
    tree = resp.json()
    AVAILABLE_FILES = [item['path'].split('/')[-1] for item in tree if item['path'].endswith('.parquet')]
except Exception as e:
    print(f"⚠️ Warning: Could not fetch index tree from HF: {e}")
    AVAILABLE_FILES = []

# DUAL ROUTING MATRICES
PHONE_PREFIXES = ["6", "7", "8", "9", "other"]
AADHAR_PREFIXES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

PHONE_MAP: dict[str, list[str]] = {p: [] for p in PHONE_PREFIXES}
AADHAR_MAP: dict[str, list[str]] = {p: [] for p in AADHAR_PREFIXES}
ALL_URLS: list[str] = []

for filename in AVAILABLE_FILES:
    url = f"{HF_INDEX_BASE}/{filename}"
    ALL_URLS.append(url)
    
    try:
        prefix = filename.split('_')[-1].split('.')[0]
        if "addhar" in filename.lower() or "aadhar" in filename.lower():
            if prefix in AADHAR_MAP: AADHAR_MAP[prefix].append(url)
        else:
            if prefix in PHONE_MAP: PHONE_MAP[prefix].append(url)
            else: PHONE_MAP["other"].append(url)
    except Exception:
        pass

print(f"✅ Discovered {sum(len(v) for v in PHONE_MAP.values())} Phone shards and {sum(len(v) for v in AADHAR_MAP.values())} Aadhaar shards.")

# ── 4. Global Shared DuckDB Engine (Dynamic RAM Allocation) ─────────────────
global_db = duckdb.connect(":memory:")
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck_worker")

def init_global_db():
    print("⚙️ Initializing Global DuckDB Engine with Dual Matrix Routing...")
    
    # Render Safe RAM Allocation (75% of exact available memory)
    total_ram_mb = int(psutil.virtual_memory().total / (1024**2))
    duckdb_ram_mb = max(256, int(total_ram_mb * 0.75))
    
    global_db.execute(f"SET home_directory='{SAFE_TEMP}'")
    global_db.execute("INSTALL parquet; LOAD parquet;")
    global_db.execute("INSTALL httpfs; LOAD httpfs;")
    
    global_db.execute("SET enable_object_cache=true;")
    global_db.execute("SET enable_http_metadata_cache=true;")
    global_db.execute("SET http_keep_alive=true;")
    global_db.execute("SET http_retries=5;")
    global_db.execute("SET http_retry_wait_ms=1000;")
    global_db.execute("SET http_timeout=60000;")
    global_db.execute(f"SET memory_limit='{duckdb_ram_mb}MB';")
    global_db.execute("SET preserve_insertion_order=false;")
    global_db.execute(f"SET threads={PARALLELISM};")

    for prefix, urls in PHONE_MAP.items():
        if urls:
            lst = ", ".join(f"'{u}'" for u in urls)
            global_db.execute(f"CREATE OR REPLACE VIEW phone_prefix_{prefix} AS SELECT * FROM read_parquet([{lst}])")
            
    for prefix, urls in AADHAR_MAP.items():
        if urls:
            lst = ", ".join(f"'{u}'" for u in urls)
            global_db.execute(f"CREATE OR REPLACE VIEW aadhar_prefix_{prefix} AS SELECT * FROM read_parquet([{lst}])")
            
    if ALL_URLS:
        lst = ", ".join(f"'{u}'" for u in ALL_URLS)
        global_db.execute(f"CREATE OR REPLACE VIEW people_all AS SELECT * FROM read_parquet([{lst}])")

init_global_db()

# ── 5. Record Deduplication ─────────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad: return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()

def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None: continue
        value = str(raw).strip()
        if not value or value in seen: continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out

# ── 6. Search Engine Logic (Matrix Routed) ──────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS: raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''").strip()
    cache_key = f"field:{field}:{v}:{mode}:{limit}"
    
    cached = MEM_CACHE.get(cache_key)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    if mode != "exact" or not v:
        res = {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        MEM_CACHE.set(cache_key, res)
        return res

    if field == "phoneNumber":
        prefix = v[0] if v[0] in PHONE_PREFIXES else "other"
        if not PHONE_MAP.get(prefix): return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        view = f"phone_prefix_{prefix}"
    elif field == "aadharNumber":
        prefix = v[0] if v[0] in AADHAR_PREFIXES else None
        if not prefix or not AADHAR_MAP.get(prefix): return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        view = f"aadhar_prefix_{prefix}"
    else:
        view = "people_all"

    sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    
    cursor = global_db.cursor()
    try:
        rows = cursor.execute(sql).fetchall()
        cols = [d[0] for d in cursor.description]
        results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
        res = {"field": field, "value": value, "mode": mode, "count": len(results), "results": results, "from_cache": False}
        MEM_CACHE.set(cache_key, res)
        return res
    except Exception as e:
        print(f"❌ DuckDB Query Error: {e}")
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
    finally:
        cursor.close()

def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    cache_key = f"unified:{q}:{limit}"
    cached = MEM_CACHE.get(cache_key)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    is_num = q.isdigit() and len(q) >= 8
    if is_num:
        all_rows, searched = [], []
        r = _run_field_search("phoneNumber", q, "exact", limit)
        if r["results"]:
            all_rows.extend(r["results"])
            searched.append("phoneNumber")
            
        if not all_rows and len(q) >= 12:
            r = _run_field_search("aadharNumber", q, "exact", limit)
            if r["results"]:
                all_rows.extend(r["results"])
                searched.append("aadharNumber")
                
        all_rows = _cap_duplicates(all_rows)[:limit]
        res = {"query": q, "searched_fields": searched, "count": len(all_rows), "results": all_rows, "from_cache": False}
        MEM_CACHE.set(cache_key, res)
        return res
    else:
        res = {"query": q, "searched_fields": [], "count": 0, "results": [], "from_cache": False}
        MEM_CACHE.set(cache_key, res)
        return res

# ── 7. Cloudflare Tunnel Auto-Manager ───────────────────────────────────────
CLOUDFLARE_URL = None

def setup_and_run_cloudflared(port: int):
    global CLOUDFLARE_URL
    binary = shutil.which("cloudflared")
    if not binary:
        local_bin = os.path.join(BASE, "cloudflared")
        if not os.path.exists(local_bin):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            urllib.request.urlretrieve(url, local_bin)
            st = os.stat(local_bin)
            os.chmod(local_bin, st.st_mode | stat.S_IEXEC)
        binary = local_bin

    cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    for line in iter(proc.stdout.readline, ""):
        match = url_pattern.search(line)
        if match:
            CLOUDFLARE_URL = match.group(0)
            break

# ── 8. Lifecycle & FastAPI App ──────────────────────────────────────────────
async def pinger():
    url = f"http://127.0.0.1:{PORT}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try: await client.get(url)
            except Exception: pass

async def warmup_cache():
    if not ALL_URLS: return
    try:
        loop = asyncio.get_running_loop()
        def _warmup_query():
            cursor = global_db.cursor()
            for prefix in PHONE_PREFIXES:
                if PHONE_MAP.get(prefix):
                    try: cursor.execute(f"SELECT phoneNumber FROM phone_prefix_{prefix} LIMIT 1").fetchall()
                    except: pass
            for prefix in AADHAR_PREFIXES:
                if AADHAR_MAP.get(prefix):
                    try: cursor.execute(f"SELECT aadharNumber FROM aadhar_prefix_{prefix} LIMIT 1").fetchall()
                    except: pass
            cursor.close()
            
        await loop.run_in_executor(pool, _warmup_query)
        print("\n🚀 WARMUP SUCCESSFUL: Matrix Footers securely cached in RAM.\n")
    except Exception as e: pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    ping_task = asyncio.create_task(pinger())
    warmup_task = asyncio.create_task(warmup_cache())
    yield
    ping_task.cancel()
    warmup_task.cancel()
    global_db.close()

fastapi_app = FastAPI(title="ICMR + HITEK Search API", lifespan=lifespan)

@fastapi_app.get("/")
def root():
    return {"app": "ICMR + HITEK API", "active_chunks": len(ALL_URLS)}

@fastapi_app.get("/health")
def health():
    return {"status": "ok"}

@fastapi_app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    q_val = (q or mobile or "").strip()
    if not q_val: raise HTTPException(422, "Provide q or mobile")
    
    start_time = time.perf_counter()
    loop = asyncio.get_running_loop()
    
    if field: data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else: data = await loop.run_in_executor(pool, _unified_search, q_val, limit)
    
    duration = time.perf_counter() - start_time
    
    result = {
        "success": bool(data["count"]),
        "time_taken_seconds": round(duration, 4),
        "from_cache": data.get("from_cache", False),
        **data,
        "number": q_val,
        "total": data["count"]
    }
    
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")

# ── 9. Gradio Interface ─────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val: lines.append(f"**{field}:** {val}")
    return "\n\n".join(lines)

def search_ui(query: str, limit: int) -> str:
    if not query.strip(): return "⚠️ Please enter a search query."
    
    start_time = time.perf_counter()
    try:
        data = _unified_search(query.strip(), int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    duration = time.perf_counter() - start_time
    
    cache_badge = "⚡ (Served from RAM)" if data.get("from_cache") else ""
    
    if not data["results"]:
        return f"❌ **No data found** for `{query}`. (Time: {duration:.3f}s)"
        
    parts = [f"🔍 **Query:** `{query}`  |  **Found:** {data['count']} results  |  ⏱️ **Time:** {duration:.3f}s {cache_badge}\n\n---"]
    for i, row in enumerate(data["results"], 1):
        parts.append(f"### Result {i}\n{format_result(row)}\n\n---")
    return "\n".join(parts)

def build_ui():
    with gr.Blocks(title="ICMR Search API") as demo:
        gr.Markdown("# ⚡ ICMR + HITEK Search API")
        with gr.Row():
            with gr.Column(scale=3): query_input = gr.Textbox(label="Search Query", placeholder="Enter query...", lines=1)
            with gr.Column(scale=1): limit_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    
    # Check if running on Render.com
    IS_RENDER = os.environ.get("RENDER") == "true"
    RENDER_URL = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    
    # Only boot Cloudflare if running Locally (Saves RAM/CPU on Render)
    if not IS_RENDER:
        threading.Thread(target=setup_and_run_cloudflared, args=(PORT,), daemon=True).start()
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_lan_ip = s.getsockname()[0]
        s.close()
    except: local_lan_ip = "127.0.0.1"
        
    print("\n" + "="*80)
    if IS_RENDER and RENDER_URL:
        print("🚀 RENDER CLOUD HOSTING DETECTED (Ultra-Fast Mode Active)")
        print(f"🌍 1. Public Web UI   : https://{RENDER_URL}/")
        print(f"🤖 2. Public API      : https://{RENDER_URL}/search?q=6203913021")
    else:
        print("🚀 LOCAL TESTING LINKS (Ctrl+Click in VS Code):")
        print(f"🏠 1. Localhost UI   : http://127.0.0.1:{PORT}/")
        print(f"🤖 2. Localhost API  : http://127.0.0.1:{PORT}/search?q=6203913021")
        print("-" * 80)
        print(f"📱 3. Network UI     : http://{local_lan_ip}:{PORT}/  <-- (Test on your phone via WiFi)")
        
        time.sleep(3) # Wait for Cloudflare
        if CLOUDFLARE_URL: print(f"☁️ 4. Cloudflare URL : {CLOUDFLARE_URL}/")
        
    print("="*80 + "\n")
    
    # uvicorn handles the startup port binding smoothly now that asyncio is imported
    uvicorn.run(app, host="0.0.0.0", port=PORT)
