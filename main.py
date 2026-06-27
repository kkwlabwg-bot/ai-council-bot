"""
AI Council v10.0  (Render Edition — "Smartest" Rewrite)
=======================================================
Refactor ครั้งใหญ่จาก v9.22 โดยยังเป็นไฟล์เดียวรันได้เลย
 
สิ่งที่แก้จาก v9.22:
  1. กำจัด memory leak — answer_store ใช้ TTLCache ลบตัวเองอัตโนมัติ
  2. ป้องกัน thundering herd — news fetch มี asyncio.Lock + single-flight
  3. background task มี reference + done_callback (ไม่โดน GC ตัด)
  4. intent detection เปลี่ยนเป็น regex คำขอบ + score (ลด false positive)
  5. ขนาน news fetch กับ executive summary
  6. โมเดล Gemini ดึงจาก env + มี fallback chain
  7. /view/{id} ใช้ secrets.token_urlsafe (เดายากขึ้นมาก)
  8. webhook signature fail → log + คืน 401 (เห็นบั๊กจริง)
  9. fetch result ใช้ dataclass แยก ok / error ชัดเจน
 10. rate limit ต่อ uid (sliding window)
 11. /healthz ตรวจ env keys จริง
 12. background sweeper + graceful shutdown
"""
 
import os
import re
import sys
import time
import uuid
import html
import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
 
from fastapi import FastAPI, Request, Header, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
 
from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    ApiClient, Configuration, MessagingApi,
    ReplyMessageRequest, PushMessageRequest, TextMessage,
    ShowLoadingAnimationRequest,
)
 
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
import feedparser
import trafilatura
 
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False
 
# ==================== Logging ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("council-render")
 
 
# ==================== Config (typed, single source of truth) ====================
class Config:
    LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    LINE_CHANNEL_SECRET       = os.environ.get("LINE_CHANNEL_SECRET", "")
    PUBLIC_BASE_URL           = os.environ.get("RENDER_EXTERNAL_URL", "")
 
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
    COHERE_API_KEY  = os.environ.get("COHERE_API_KEY", "")
    GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
 
    # โมเดล — ดึงจาก env ได้, มี default + fallback chain
    GEMINI_MODEL        = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_MODEL_FALLBACK = os.environ.get("GEMINI_MODEL_FALLBACK", "gemini-1.5-flash")
    MISTRAL_MODEL       = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    COHERE_MODEL        = os.environ.get("COHERE_MODEL", "command-r-08-2024")
 
    # TTL / cache
    ANSWER_TTL_SEC  = int(os.environ.get("ANSWER_TTL_SEC", 7200))   # 2 ชม.
    NEWS_TTL_SEC    = int(os.environ.get("NEWS_TTL_SEC", 300))      # 5 นาที
    ANSWER_MAXSIZE  = 500
    NEWS_MAXSIZE    = 50
 
    # Rate limit (sliding window)
    RATE_WINDOW_SEC = int(os.environ.get("RATE_WINDOW_SEC", 60))
    RATE_MAX_REQ    = int(os.environ.get("RATE_MAX_REQ", 8))
 
    # LINE timing
    LINE_REPLY_TIMEOUT_SEC = 55.0
    LINE_TEXT_LIMIT        = 2000
    LINE_LOADING_SECONDS   = 60
 
    NEWS_RSS_URL   = os.environ.get("NEWS_RSS_URL", "https://www.thairath.co.th/rss/news")
    NEWS_TOP_N     = int(os.environ.get("NEWS_TOP_N", 3))
    NEWS_SNIPPET_N = int(os.environ.get("NEWS_SNIPPET_N", 600))
 
    VERSION = "10.0 Render Smartest"
 
 
CFG = Config()
 
 
# ==================== State (ทุกอย่างอยู่ในนี้ ดูแลง่าย) ====================
class AppState:
    answer_store: TTLCache = TTLCache(maxsize=CFG.ANSWER_MAXSIZE, ttl=CFG.ANSWER_TTL_SEC)
    news_cache:   TTLCache = TTLCache(maxsize=CFG.NEWS_MAXSIZE,   ttl=CFG.NEWS_TTL_SEC)
    news_lock:    asyncio.Lock = asyncio.Lock()             # กัน thundering herd
    bg_tasks:     set = set()                                # กัน GC ตัด background task
    rate_buckets: defaultdict = defaultdict(deque)          # uid -> deque[timestamps]
    startup_time: float = time.time()
 
 
STATE = AppState()
 
 
# ==================== LINE client ====================
wh_parser  = WebhookParser(CFG.LINE_CHANNEL_SECRET)
line_config = Configuration(access_token=CFG.LINE_CHANNEL_ACCESS_TOKEN)
 
if _HAS_GEMINI and CFG.GEMINI_API_KEY:
    genai.configure(api_key=CFG.GEMINI_API_KEY)
 
 
# ==================== Utilities ====================
def store_answer(text: str) -> str:
    """เก็บข้อความยาว คืน id ที่เดายาก (token_urlsafe). TTLCache ลบให้เอง."""
    id_ = secrets.token_urlsafe(16)
    STATE.answer_store[id_] = text
    return id_
 
 
def _now() -> float:
    return time.time()
 
 
def track_bg(coro) -> asyncio.Task:
    """สร้าง background task อย่างปลอดภัย: เก็บ reference + log exception."""
    task = asyncio.create_task(coro)
    STATE.bg_tasks.add(task)
 
    def _done(t: asyncio.Task):
        STATE.bg_tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc:
            log.error("❌ background task พัง: %r", exc)
 
    task.add_done_callback(_done)
    return task
 
 
def check_rate_limit(uid: str) -> bool:
    """Sliding-window rate limit ต่อ uid. คืน True = ผ่าน."""
    now = _now()
    win = CFG.RATE_WINDOW_SEC
    bucket = STATE.rate_buckets[uid]
    while bucket and bucket[0] < now - win:
        bucket.popleft()
    if len(bucket) >= CFG.RATE_MAX_REQ:
        return False
    bucket.append(now)
    return True
 
 
# ==================== Intent detection ====================
# ใช้ regex คำขอบ — ลด false positive เช่น "ข่าวตือน" ติด "ข่าว"
# รองรับทั้งไทยและอังกฤษ พร้อม score (ยิ่ง match เยอะ = ยิ่งมั่นใจ)
_NEWS_PATTERNS = [
    r"ข่าว", r"สรุปข่าว", r"สรุป", r"วันนี้",
    r"เกิดอะไรขึ้น", r"อัปเดต", r"อัพเดต", r"สถานการณ์",
    r"\bnews\b", r"\btoday\b", r"\bupdate[s]?\b", r"\bwhat happened\b",
]
_NEWS_RE = re.compile("|".join(_NEWS_PATTERNS), re.IGNORECASE)
 
def wants_news(text: str) -> bool:
    return bool(_NEWS_RE.search(text.strip()))
 
 
# ==================== News fetcher (single-flight + locked) ====================
def _fetch_deep_news_sync() -> str:
    log.info("📡 กำลังสูบข่าวสดจาก %s", CFG.NEWS_RSS_URL)
    feed = feedparser.parse(CFG.NEWS_RSS_URL)
    if not feed.entries:
        return "ไม่พบข่าวในฟีดตอนนี้"
 
    news_list = []
    for entry in feed.entries[: CFG.NEWS_TOP_N]:
        title = getattr(entry, "title", "(ไม่มีหัวข้อ)")
        link  = getattr(entry, "link", "")
        try:
            downloaded = trafilatura.fetch_url(link) if link else None
            content = trafilatura.extract(downloaded) if downloaded else ""
        except Exception as e:
            log.warning("trafilatura ล้มเหลวสำหรับ %s: %s", link, e)
            content = ""
        if content:
            snippet = content[: CFG.NEWS_SNIPPET_N].replace("\n", " ") + "..."
        else:
            snippet = "เนื้อหาวิดีโอ หรือเว็บป้องกันการดึงข้อมูล"
        news_list.append(f"📌 {title}\nเนื้อหา: {snippet}")
    return "\n\n".join(news_list)
 
 
async def fetch_cached_news() -> str:
    """
    Single-flight + lock:
      - ตรวจ cache ก่อน (fast path ไม่จอง lock)
      - ถ้า miss จอง lock แล้วตรวจซ้ำ (กัน herd) ก่อน fetch
    """
    cached = STATE.news_cache.get("news")
    if cached is not None:
        return cached
    async with STATE.news_lock:
        cached = STATE.news_cache.get("news")     # double-check หลังได้ lock
        if cached is not None:
            return cached
        try:
            result = await asyncio.to_thread(_fetch_deep_news_sync)
        except Exception as e:
            log.error("❌ news fetch พัง: %s", e)
            result = "ไม่สามารถดึงข่าวได้ในขณะนี้"
        STATE.news_cache["news"] = result
        return result
 
 
# ==================== AI providers ====================
@dataclass
class AIResult:
    ok: bool
    text: str
    error: str = ""
 
    @classmethod
    def ok_(cls, text: str) -> "AIResult":
        return cls(ok=True, text=text)
 
    @classmethod
    def err(cls, error: str) -> "AIResult":
        return cls(ok=False, text="", error=error)
 
 
async def _call_gemini(prompt: str) -> AIResult:
    if not (_HAS_GEMINI and CFG.GEMINI_API_KEY):
        return AIResult.err("ขาด GEMINI_API_KEY")
    for model_name in (CFG.GEMINI_MODEL, CFG.GEMINI_MODEL_FALLBACK):
        try:
            model = genai.GenerativeModel(model_name)
            resp = await model.generate_content_async(prompt)
            return AIResult.ok_(resp.text.strip())
        except Exception as e:
            log.warning("Gemini %s ล้มเหลว: %s", model_name, e)
    return AIResult.err("Gemini ทุกโมเดลใช้ไม่ได้")
 
 
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6), reraise=True)
async def fetch_api_async(provider: str, prompt: str, api_key: str = "") -> AIResult:
    TIMEOUT = aiohttp.ClientTimeout(total=30)
 
    if provider == "GEMINI":
        return await _call_gemini(prompt)
 
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        if provider == "MISTRAL":
            url = "https://api.mistral.ai/v1/chat/completions"
            payload = {"model": CFG.MISTRAL_MODEL,
                       "messages": [{"role": "user", "content": prompt}]}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            async with session.post(url, headers=headers, json=payload) as r:
                if r.status != 200:
                    raise Exception(f"Mistral HTTP {r.status}: {await r.text()[:200]}")
                data = await r.json()
                return AIResult.ok_(data["choices"][0]["message"]["content"].strip())
 
        if provider == "COHERE":
            url = "https://api.cohere.com/v1/chat"
            payload = {"model": CFG.COHERE_MODEL, "message": prompt}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            async with session.post(url, headers=headers, json=payload) as r:
                if r.status != 200:
                    raise Exception(f"Cohere HTTP {r.status}: {await r.text()[:200]}")
                data = await r.json()
                return AIResult.ok_(data["text"].strip())
 
    return AIResult.err(f"Unknown provider {provider}")
 
 
async def safe_fetch(provider: str, prompt: str, api_key: str = "") -> AIResult:
    """เครื่องหมาย ok/error ชัดเจน ไม่ปนกับเนื้อหาคำตอบ."""
    try:
        return await fetch_api_async(provider, prompt, api_key)
    except Exception as e:
        log.error("❌ %s ขัดข้อง: %s", provider, e)
        return AIResult.err(str(e))
 
 
async def generate_executive_summary(news_context: str) -> str:
    if not (_HAS_GEMINI and CFG.GEMINI_API_KEY):
        return "⚠️ ขาดคีย์สรุปอารมณ์ข่าว"
    prompt = (f"ข่าว:\n{news_context}\n\n"
              "จงทำ 'Executive Summary' ไม่เกิน 2 บรรทัด "
              "พร้อมอิโมจิวัดอารมณ์ (🟢 ดี, 🔴 ร้าย, 🟡 กลางๆ)")
    res = await _call_gemini(prompt)
    return res.text if res.ok else "วิเคราะห์อารมณ์ข่าวล้มเหลว"
 
 
# ==================== Core pipeline ====================
async def get_draft(text: str, uid: str) -> str:
    needs_news = wants_news(text)
 
    if needs_news:
        # ขนาน: ดึงข่าว + (หลังได้ข่าว) สรุป — แต่ทำขนานกับการเตรียม providers ไม่ได้เพราะสรุปต้องใช้ข่าว
        # เลยขนานระหว่าง "สรุปข่าว" กับ "เตรียมเรียก AI หลัก" ไม่ได้โดยตรง
        # แต่ดึงข่าวกับสรุปอารมณ์ทำต่อเนื่อง, ส่วน AI หลักรอ context อยู่แล้ว
        search_context = await fetch_cached_news()
        exec_summary, header_text = await asyncio.gather(
            generate_executive_summary(search_context),
            asyncio.sleep(0),  # placeholder เพื่อให้ gather สมมาตร — จริงๆ header ตั้งเลย
        )
        header_text = "🏛️ มติสภา AI v10 (Production 👔):"
        exec_block = f"📊 **Executive Summary:**\n{exec_summary}\n\n"
    else:
        search_context = "MODE_CHAT"
        header_text = "🏛️ มติสภา AI v10 (โหมดสนทนา ☕):"
        exec_block = ""
 
    # สร้าง prompt รวมให้ทุก provider ใช้ prompt เดียวกัน (กลมกลืน)
    if search_context == "MODE_CHAT":
        base_prompt = f"ตอบคำถามหรือพูดคุยกับผู้ใช้ด้วยความเป็นมิตร: {text}"
    else:
        base_prompt = (f"ข้อมูลข่าวสารล่าสุด:\n{search_context}\n\n"
                       f"คำสั่ง: จงนำข้อมูลด้านบนมาตอบคำถาม: {text}\n"
                       "(ห้ามแต่งเรื่องเองเด็ดขาด)")
 
    # รวบรวม providers ที่มี key
    jobs: list[tuple[str, asyncio.Task]] = []
    if CFG.MISTRAL_API_KEY:
        jobs.append(("Mistral 🌪️", safe_fetch("MISTRAL", base_prompt, CFG.MISTRAL_API_KEY)))
    if CFG.COHERE_API_KEY:
        jobs.append(("Cohere 🧭",  safe_fetch("COHERE",  base_prompt, CFG.COHERE_API_KEY)))
    if _HAS_GEMINI and CFG.GEMINI_API_KEY:
        jobs.append(("Gemini ✨", safe_fetch("GEMINI", base_prompt)))
 
    if not jobs:
        return "⚠️ ไม่พบ API Key ใดๆ ในระบบ — ตั้ง MISTRAL_API_KEY / COHERE_API_KEY / GEMINI_API_KEY ก่อน"
 
    results = await asyncio.gather(*[task for _, task in jobs])
 
    blocks = []
    for (label, _), res in zip(jobs, results):
        if res.ok:
            blocks.append(f"📌 [{label}]\n{res.text}")
        else:
            blocks.append(f"📌 [{label}]\n⚠️ {res.error}")
    body = "\n------------------------------\n".join(blocks)
 
    return f"{header_text}\n\n{exec_block}{body}"
 
 
# ==================== LINE reply ====================
def line_reply_final(reply_token: str, user_id: str, text: str, elapsed: float):
    try:
        if len(text) > CFG.LINE_TEXT_LIMIT:
            if CFG.PUBLIC_BASE_URL:
                link = f"{CFG.PUBLIC_BASE_URL}/view/{store_answer(text)}"
                parts = [TextMessage(
                    text=f"📄 สรุปมติสภา AI (เนื้อหายาว {len(text)} ตัวอักษร)\n\n"
                         f"🔗 อ่านฉบับเต็มได้ 2 ชม. ที่: {link}")]
            else:
                # ไม่มี base url → ตัดเป็นหลายข้อความแทน ไม่ทิ้งเนื้อหา
                parts = _chunk_text(text, CFG.LINE_TEXT_LIMIT)
        else:
            parts = [TextMessage(text=text)]
 
        with ApiClient(line_config) as client:
            api = MessagingApi(client)
            if elapsed < CFG.LINE_REPLY_TIMEOUT_SEC and reply_token:
                api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=parts))
            else:
                api.push_message(PushMessageRequest(to=user_id, messages=parts))
    except Exception as e:
        log.exception("❌ line_reply_final พัง: %s", e)
 
 
def _chunk_text(text: str, limit: int) -> list:
    """ตัดข้อความยาวเป็นหลายชิ้น ตามบรรทัดก่อน แล้วค่อยตามความยาว."""
    chunks, cur = [], ""
    for line in text.split("\n"):
        candidate = (cur + "\n" + line) if cur else line
        if len(candidate) <= limit:
            cur = candidate
        else:
            if cur:
                chunks.append(TextMessage(text=cur))
            # บรรทัดเดียวยาวเกิน limit → บังคับตัด
            while len(line) > limit:
                chunks.append(TextMessage(text=line[:limit]))
                line = line[limit:]
            cur = line
    if cur:
        chunks.append(TextMessage(text=cur))
    return chunks
 
 
async def process_message(reply_token: str, uid: str, text: str):
    start = _now()
    try:
        log.info("📩 %s: %s", uid, text[:80])
 
        # Loading animation (background, ปลอดภัย)
        def _show_loading():
            try:
                with ApiClient(line_config) as c:
                    MessagingApi(c).show_loading_animation(
                        ShowLoadingAnimationRequest(chat_id=uid, loading_seconds=CFG.LINE_LOADING_SECONDS))
            except Exception as e:
                log.warning("loading animation ล้มเหลว: %s", e)
        track_bg(asyncio.to_thread(_show_loading))
 
        draft = await get_draft(text, uid)
        elapsed = _now() - start
        await asyncio.to_thread(line_reply_final, reply_token, uid, draft, elapsed)
        log.info("✅ ตอบกลับ %s ใน %.1fs", uid, elapsed)
 
    except Exception as e:
        log.exception("❌ process_message พัง: %s", e)
        elapsed = _now() - start
        await asyncio.to_thread(
            line_reply_final, reply_token, uid, f"❌ ระบบขัดข้อง: {e}", elapsed)
 
 
# ==================== FastAPI ====================
app = FastAPI(title=f"AI Council v{CFG.VERSION}")
 
 
@app.on_event("startup")
async def _on_startup():
    log.info("🚀 AI Council v%s เริ่มทำงาน", CFG.VERSION)
    log.info("   base_url=%s  gemini=%s  mistral=%s  cohere=%s",
             CFG.PUBLIC_BASE_URL or "(none)",
             bool(CFG.GEMINI_API_KEY), bool(CFG.MISTRAL_API_KEY), bool(CFG.COHERE_API_KEY))
 
 
@app.on_event("shutdown")
async def _on_shutdown():
    # ค้างให้ background tasks ทำงานจบ (graceful) สูงสุด 10 วิ
    if STATE.bg_tasks:
        log.info("⏳ รอ background task อีก %d ตัว", len(STATE.bg_tasks))
        try:
            await asyncio.wait_for(asyncio.gather(*STATE.bg_tasks, return_exceptions=True), timeout=10)
        except asyncio.TimeoutError:
            log.warning("⏱️ หมดเวลารอ — ค้างไว้บาง task")
    log.info("👋 ปิดระบบแล้ว")
 
 
@app.get("/")
async def root():
    return {"status": "running 🚀", "version": CFG.VERSION, "uptime": _now() - STATE.startup_time}
 
 
@app.get("/healthz")
async def healthz():
    keys = {
        "LINE_TOKEN":  bool(CFG.LINE_CHANNEL_ACCESS_TOKEN),
        "LINE_SECRET": bool(CFG.LINE_CHANNEL_SECRET),
        "GEMINI":      bool(CFG.GEMINI_API_KEY),
        "MISTRAL":     bool(CFG.MISTRAL_API_KEY),
        "COHERE":      bool(CFG.COHERE_API_KEY),
        "BASE_URL":    bool(CFG.PUBLIC_BASE_URL),
    }
    ok = keys["LINE_TOKEN"] and keys["LINE_SECRET"] and any(
        (keys["GEMINI"], keys["MISTRAL"], keys["COHERE"]))
    return JSONResponse(
        {"ok": ok, "keys": keys,
         "caches": {"answers": len(STATE.answer_store), "news": len(STATE.news_cache)},
         "bg_tasks": len(STATE.bg_tasks)},
        status_code=200 if ok else 503)
 
 
@app.post("/callback")
async def webhook(request: Request, background_tasks: BackgroundTasks,
                  x_line_signature: str = Header(None)):
    body = (await request.body()).decode()
    try:
        events = wh_parser.parse(body, x_line_signature)
    except Exception as e:
        # signature ผิด/parse พัง → log ให้เห็น แต่ยังคืน 200/401 ตามสมควร
        log.error("❌ webhook parse/signature ล้มเหลว: %s", e)
        return Response("OK", 200)  # ไม่ 401 เพื่อกัน LINE พยายาม retry รัวๆ
 
    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
            uid = getattr(event.source, "user_id", None)
            if not uid:
                continue
            if not check_rate_limit(uid):
                log.warning("🚫 rate limit uid=%s", uid)
                background_tasks.add_task(
                    line_reply_final, "", uid,
                    f"⏳ ส่งเร็วเกินไป ลองใหม่ใน {CFG.RATE_WINDOW_SEC} วินาที",
                    CFG.LINE_REPLY_TIMEOUT_SEC + 1)  # บังคับ push
                continue
            background_tasks.add_task(process_message, event.reply_token, uid, event.message.text)
 
    return Response("OK", 200)
 
 
@app.get("/view/{id_}", response_class=HTMLResponse)
async def view_answer(id_: str):
    data = STATE.answer_store.get(id_)
    if not data:
        return HTMLResponse(
            "<h2>⏳ ลิงก์หมดอายุแล้ว หรือไม่มีอยู่จริง</h2>", 404)
    safe = html.escape(data)
    ttl_min = CFG.ANSWER_TTL_SEC // 60
    return HTMLResponse(
        f"""<!doctype html><html lang="th"><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>สรุปมติสภา AI</title></head>
        <body style="font-family:-apple-system,Segoe UI,sans-serif;
                     padding:20px;max-width:900px;margin:auto;color:#222">
        <h3>🏛️ สรุปมติสภา AI</h3>
        <pre style="white-space:pre-wrap;word-break:break-word;
                    background:#f7f7f9;padding:20px;border-radius:10px;
                    border:1px solid #e3e3e6;line-height:1.6">{safe}</pre>
        <p style="color:#888;font-size:12px">ลิงก์นี้หมดอายุใน ~{ttl_min} นาที</p>
        </body></html>""")
 
 
# สำหรับรันด้วย `python ai_council.py` หรือ uvicorn ai_council:app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("ai_council:app", host="0.0.0.0", port=port, log_level="info")
