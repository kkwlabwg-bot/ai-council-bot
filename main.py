import os, time, asyncio, logging, uuid, html, sys
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Header, Response, BackgroundTasks
from fastapi.responses import HTMLResponse

from linebot.v3.webhook import WebhookParser
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    ApiClient, Configuration, MessagingApi,
    ReplyMessageRequest, PushMessageRequest, TextMessage,
    ShowLoadingAnimationRequest,
)

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from cachetools import TTLCache, cached
import feedparser
import trafilatura
import google.generativeai as genai

# ==================== ตั้งค่าระบบ Logging ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger("council-render")

# ==================== ตั้งค่า Environment Variables ====================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
PUBLIC_BASE_URL = os.environ.get("RENDER_EXTERNAL_URL", "") # Render จะสร้างให้เอง

app = FastAPI(title="AI Council v9.22 (Render Edition)")
wh_parser = WebhookParser(LINE_CHANNEL_SECRET)
line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

answer_store = {}
news_cache = TTLCache(maxsize=100, ttl=300)

if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# ==================== ระบบจัดการข้อความ ====================
def store_answer(text: str) -> str:
    id = uuid.uuid4().hex[:10]
    answer_store[id] = {"text": text, "expire": datetime.now() + timedelta(minutes=120)}
    return id

def line_reply_final(reply_token: str, user_id: str, text: str, elapsed: float):
    try:
        if len(text) > 2000:
            link = f"{PUBLIC_BASE_URL}/view/{store_answer(text)}" if PUBLIC_BASE_URL else "เนื้อหายาวเกินไป"
            parts = [TextMessage(text=f"📄 สรุปมติสภา AI (เนื้อหายาวเกินไป)\n\n🔗 คลิกเปิดอ่านฉบับเต็มที่นี่: {link}")]
        else:
            parts = [TextMessage(text=text)]

        with ApiClient(line_config) as client:
            api = MessagingApi(client)
            # ถ้านานกว่า 55 วินาที LINE จะถือว่า Time Out ให้ใช้ Push Message แทน
            if elapsed < 55:
                api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=parts))
            else:
                api.push_message(PushMessageRequest(to=user_id, messages=parts))
    except Exception as e:
        log.exception(f"❌ ระบบส่งข้อความกลับพัง (line_reply_final): {e}")

# ==================== ระบบดึงข่าว ====================
@cached(cache=news_cache)
def fetch_deep_news_sync() -> str:
    try:
        log.info("📡 กำลังสูบข้อมูลข่าวสดใหม่...")
        feed = feedparser.parse("https://www.thairath.co.th/rss/news")
        news_list = []
        for entry in feed.entries[:3]:
            title = entry.title
            link = entry.link
            downloaded = trafilatura.fetch_url(link)
            content = trafilatura.extract(downloaded) if downloaded else ""
            snippet = content[:600].replace('\n', ' ') + "..." if content else "เนื้อหาวิดีโอหรือหน้าเว็บป้องกันการดึงข้อมูล"
            news_list.append(f"📌 {title}\nเนื้อหา: {snippet}")
        return "\n\n".join(news_list)
    except Exception as e:
        log.error(f"❌ Scraper error: {e}")
        return "ไม่สามารถดึงข้อมูลข่าวสารได้"

async def fetch_cached_news():
    return await asyncio.to_thread(fetch_deep_news_sync)

# ==================== ระบบเชื่อมต่อ AI 3 ค่าย ====================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
async def fetch_api_async(provider, text, search_context, api_key=None):
    TIMEOUT = aiohttp.ClientTimeout(total=30)
    
    if search_context == "MODE_CHAT":
        prompt = f"ตอบคำถามหรือพูดคุยกับผู้ใช้ด้วยความเป็นมิตร: {text}"
    else:
        prompt = f"ข้อมูลข่าวสารล่าสุด:\n{search_context}\n\nคำสั่ง: จงนำข้อมูลด้านบนมาตอบคำถาม: {text}\n(ห้ามแต่งเรื่องเองเด็ดขาด)"
    
    if provider == "GEMINI":
        if not os.environ.get("GEMINI_API_KEY"): return "⚠️ ไม่มี API Key"
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await model.generate_content_async(prompt)
        return response.text.strip()

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        if provider == "MISTRAL":
            url = "https://api.mistral.ai/v1/chat/completions"
            payload = {"model": "mistral-small-latest", "messages": [{"role": "user", "content": prompt}]}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            async with session.post(url, headers=headers, json=payload) as r:
                if r.status != 200: raise Exception(f"Mistral HTTP {r.status}")
                return (await r.json())["choices"][0]["message"]["content"].strip()

        elif provider == "COHERE":
            url = "https://api.cohere.com/v1/chat"
            payload = {"model": "command-r-08-2024", "message": prompt}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            async with session.post(url, headers=headers, json=payload) as r:
                if r.status != 200: raise Exception(f"Cohere HTTP {r.status}")
                return (await r.json())["text"].strip()

async def generate_executive_summary(news_context):
    try:
        if not os.environ.get("GEMINI_API_KEY"): return "⚠️ ขาดคีย์สรุป"
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"ข่าว:\n{news_context}\n\nจงทำ 'Executive Summary' ไม่เกิน 2 บรรทัด พร้อมอิโมจิวัดอารมณ์ (🟢 ดี, 🔴 ร้าย, 🟡 กลางๆ)"
        res = await model.generate_content_async(prompt)
        return res.text.strip()
    except Exception: 
        return "วิเคราะห์อารมณ์ข่าวล้มเหลว"

# ==================== ระบบจัดรูปหน้ากระดาษ (Core Logic) ====================
async def get_draft(text, uid) -> tuple[str, str]:
    tasks, providers = [], []
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    cohere_key = os.environ.get("COHERE_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    news_keywords = ["ข่าว", "สรุป", "วันนี้", "เกิดอะไร", "อัปเดต", "สถานการณ์"]
    needs_news = any(word in text.lower() for word in news_keywords)
    
    executive_sum = ""
    if needs_news:
        search_context = await fetch_cached_news()
        header_text = "🏛️ มติสภา AI v9.22 (Production 👔):"
        executive_sum = f"📊 **Executive Summary:**\n{await generate_executive_summary(search_context)}\n\n"
    else:
        search_context = "MODE_CHAT"
        header_text = "🏛️ มติสภา AI v9.22 (โหมดสนทนา ☕):"

    async def safe_fetch(prov, txt, ctx, key=None):
        try: return await fetch_api_async(prov, txt, ctx, key)
        except Exception as e: return f"❌ ขัดข้อง: {e}"

    if mistral_key: tasks.append(safe_fetch("MISTRAL", text, search_context, mistral_key)); providers.append("Mistral 🌪️")
    if cohere_key: tasks.append(safe_fetch("COHERE", text, search_context, cohere_key)); providers.append("Cohere 🧭")
    if gemini_key: tasks.append(safe_fetch("GEMINI", text, search_context)); providers.append("Gemini ✨")

    if not tasks: return "⚠️ ไม่พบ API Key ในระบบเลย", ""
    results = await asyncio.gather(*tasks)
    return f"{header_text}\n\n{executive_sum}" + "\n".join([f"📌 [{p}]\n{ans}\n------------------------------" for p, ans in zip(providers, results)]), ""

async def process_message(reply_token, uid, text):
    start_time = time.time()
    try:
        log.info(f"📩 ได้รับข้อความจาก {uid}: {text}")
        # สั่งแสดง Loading Animation (ทำงานแบบ Background Thread)
        asyncio.create_task(asyncio.to_thread(
            lambda: MessagingApi(ApiClient(line_config)).show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=uid, loading_seconds=20)
            )
        ))
        
        draft, _ = await get_draft(text, uid)
        
        # ส่งข้อความกลับ
        await asyncio.to_thread(line_reply_final, reply_token, uid, draft, time.time() - start_time)
        log.info("✅ ตอบกลับข้อความเรียบร้อยแล้ว!")
        
    except Exception as e:
        log.error(f"❌ เกิดข้อผิดพลาดร้ายแรงใน process_message: {e}")
        await asyncio.to_thread(line_reply_final, reply_token, uid, f"❌ ระบบขัดข้อง: {e}", time.time() - start_time)

# ==================== FastAPI Endpoints ====================
@app.get("/")
async def root():
    return {"status": "AI Council is running 🚀", "version": "9.22 Render Edition (BackgroundTasks Fixed)"}

@app.post("/callback")
async def webhook(request: Request, background_tasks: BackgroundTasks, x_line_signature: str = Header(None)):
    try:
        body = (await request.body()).decode()
        events = wh_parser.parse(body, x_line_signature)
        for event in events:
            if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
                uid = getattr(event.source, "user_id", None)
                if uid: 
                    # 🌟 สำคัญ: ใช้ BackgroundTasks ตรงนี้ เพื่อให้ Render ไม่ตัดจบการทำงาน
                    background_tasks.add_task(process_message, event.reply_token, uid, event.message.text)
        return Response("OK", 200)
    except Exception as e:
        log.error(f"❌ แจ้งเตือน Webhook พัง: {e}")
        return Response("OK", 200)

@app.get("/view/{id}", response_class=HTMLResponse)
async def view_answer(id: str):
    data = answer_store.get(id)
    if not data: return HTMLResponse("<h2>⏳ ลิงก์หมดอายุแล้ว</h2>", 404)
    return HTMLResponse(f"<html><body style='font-family:sans-serif;padding:20px;max-width:900px;margin:auto;'><pre style='white-space:pre-wrap;background:#f9f9f9;padding:20px;border-radius:8px;'>{html.escape(data['text'])}</pre></body></html>")
