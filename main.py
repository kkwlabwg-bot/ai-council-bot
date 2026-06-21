import os, time, asyncio, logging, uuid, html, sys
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Header, Response
from fastapi.responses import HTMLResponse

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
from duckduckgo_search import DDGS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wellness-bot")

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
PUBLIC_BASE_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

line_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
wh_parser = WebhookParser(LINE_CHANNEL_SECRET)

user_memory = defaultdict(lambda: deque(maxlen=4))
answer_store = {}
search_cache = TTLCache(maxsize=100, ttl=300)
subscribers = set()

def sync_search(query: str):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=3, region='wt-wt'))

async def fetch_web_search(query: str) -> str:
    if query in search_cache: return search_cache[query]
    try:
        results = await asyncio.to_thread(sync_search, query)
        if not results: return "ไม่พบข้อมูลในอินเทอร์เน็ต"
        res_str = "\n".join([f"📍 {r['title']}:\n{r.get('body', r.get('snippet', ''))}" for r in results])
        search_cache[query] = res_str
        return res_str
    except Exception as e: return f"ระบบค้นหาเว็บขัดข้อง: {e}"

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=4))
async def call_llm(provider: str, prompt: str, api_key: str) -> str:
    TIMEOUT = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        if provider == "GEMINI":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            async with session.post(url, headers={"Content-Type": "application/json"}, json={"contents": [{"parts": [{"text": prompt}]}]}) as r:
                r.raise_for_status()
                return (await r.json())["candidates"][0]["content"]["parts"][0]["text"].strip()
        elif provider == "MISTRAL":
            url = "https://api.mistral.ai/v1/chat/completions"
            async with session.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "mistral-small-latest", "messages": [{"role": "user", "content": prompt}]}) as r:
                r.raise_for_status()
                return (await r.json())["choices"][0]["message"]["content"].strip()
        elif provider == "COHERE":
            url = "https://api.cohere.com/v1/chat"
            async with session.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "command-r-08-2024", "message": prompt}) as r:
                r.raise_for_status()
                return (await r.json())["text"].strip()
    return ""

async def get_raw_draft(provider, text, search_context, history, api_key):
    sys_prompt = f"[ข้อมูลค้นหาล่าสุด]:\n{search_context}\n\n[ประวัติ]:\n{history}\n\nคำสั่ง: ตอบคำถามผู้ใช้: \"{text}\""
    try: return await call_llm(provider, sys_prompt, api_key)
    except Exception: return None

async def synthesize_drafts(user_text, drafts, keys):
    drafts_text = "\n\n".join([f"--- ร่างจาก {p} ---\n{d[:2500]}" for p, d in drafts.items()])
    prompt = f"ผู้ใช้พิมพ์: \"{user_text}\"\nร่างคำตอบ:\n{drafts_text}\nคำสั่ง: สรุปคำตอบที่ดีที่สุด ใช้น้ำเสียงเป็นมิตร อบอุ่น ให้กำลังใจ และใส่ Emoji น่ารักๆ"
    for provider, key in [("GEMINI", keys.get("GEMINI")), ("MISTRAL", keys.get("MISTRAL")), ("COHERE", keys.get("COHERE"))]:
        if key:
            try: return await call_llm(provider, prompt, key)
            except: continue
    return "❌ ระบบประมวลผลล้มเหลวชั่วคราวครับ ฮึบๆ เดี๋ยวมันก็กลับมา"

async def get_consensus_answer(text, uid) -> str:
    keys = {"MISTRAL": os.environ.get("MISTRAL_API_KEY", ""), "COHERE": os.environ.get("COHERE_API_KEY", ""), "GEMINI": os.environ.get("GEMINI_API_KEY", "")}
    # เพิ่มคีย์เวิร์ดเกี่ยวกับ อากาศ และฝุ่น pm2.5 เผื่อผู้ใช้ถาม
    search_keywords = ["ข่าว", "สรุป", "วันนี้", "เกิดอะไร", "อัปเดต", "ล่าสุด", "คืออะไร", "อากาศ", "ฝนตก", "pm", "ฝุ่น"]
    needs_search = any(word in text.lower() for word in search_keywords)
    search_context = await fetch_web_search(text) if needs_search else "(ไม่ใช้เว็บ)"
    history_context = "\n".join([f"User: {q}\nAI: {a}" for q, a in user_memory[uid]]) if user_memory[uid] else "ไม่มีประวัติ"
    
    tasks, providers = [], []
    for prov, key in keys.items():
        if key:
            tasks.append(get_raw_draft(prov, text, search_context, history_context, key))
            providers.append(prov)
            
    results = await asyncio.gather(*tasks)
    valid_drafts = {p: r for p, r in zip(providers, results) if r is not None and len(r) > 5}
    if not valid_drafts: return "❌ ขออภัยครับ ตอนนี้ผมเหนื่อยนิดหน่อย (เซิร์ฟเวอร์ล่ม) เดี๋ยวขอไปพักแป๊บนะครับ 😅"
        
    final_answer = await synthesize_drafts(text, valid_drafts, keys)
    user_memory[uid].append((text, final_answer[:300] + "..."))
    return final_answer

def store_answer(text: str) -> str:
    id = uuid.uuid4().hex[:10]
    answer_store[id] = {"text": text, "expire": datetime.now() + timedelta(minutes=120)}
    return id

def line_reply_final(reply_token: str, user_id: str, text: str):
    try:
        if len(text) > 2000:
            link = f"{PUBLIC_BASE_URL}/view/{store_answer(text)}"
            parts = [TextMessage(text=f"📄 ข้อความยาวเกินไป อ่านต่อที่นี่เลยครับ:\n🔗 {link}")]
        else:
            parts = [TextMessage(text=text)]

        with ApiClient(line_config) as client:
            MessagingApi(client).reply_message(ReplyMessageRequest(reply_token=reply_token, messages=parts))
    except Exception as e: log.exception("line_reply error")

async def process_message(reply_token, uid, text):
    try:
        asyncio.create_task(asyncio.to_thread(lambda: MessagingApi(ApiClient(line_config)).show_loading_animation(ShowLoadingAnimationRequest(chat_id=uid, loading_seconds=30))))
        final_answer = await get_consensus_answer(text, uid)
        # ส่งข้อความไปเลยแบบอบอุ่น ไม่มีคำว่า มติสภา AI แล้ว
        await asyncio.to_thread(line_reply_final, reply_token, uid, final_answer)
    except Exception as e:
        await asyncio.to_thread(line_reply_final, reply_token, uid, f"❌ ขัดข้อง: {e}")

# --- Background Task (ผู้ช่วยให้กำลังใจระหว่างวัน) ---
async def proactive_wellness_routine():
    while True:
        now = datetime.now()
        # เวลาบน Render เป็น UTC ให้บวก 7 ชั่วโมงเพื่อเป็นเวลาไทย
        thai_time = now + timedelta(hours=7)
        
        # เงื่อนไข: ส่งระหว่าง 09:00 ถึง 19:00 และส่งเฉพาะนาทีที่ 00 หรือ 30
        is_valid_time = (9 <= thai_time.hour < 19 and thai_time.minute in (0, 30)) or (thai_time.hour == 19 and thai_time.minute == 0)
        
        if is_valid_time:
            if subscribers:
                time_str = thai_time.strftime("%H:%M")
                prompt = f"""
ตอนนี้เวลา {time_str} น. ในกรุงเทพฯ
คุณคือบอตผู้ช่วยส่วนตัวที่คอยดูแลหัวใจและสุขภาพ (Wellness Assistant)
เขียนข้อความ 3-4 บรรทัด เพื่อส่งให้ผู้ใช้งาน
เนื้อหา: ทักทายตามช่วงเวลา, ให้กำลังใจในการทำงาน/เรียน, เตือนให้ดูแลสุขภาพ (เช่น ดื่มน้ำ พักสายตา) หรือบอกทริคเล็กๆ สำหรับชีวิตในกรุงเทพฯ
ข้อห้าม: ห้ามเขียนยาว ห้ามทางการ ใช้น้ำเสียงเป็นกันเอง น่ารัก และมี Emoji ประกอบ
"""
                answer = await get_consensus_answer(prompt, "SYSTEM_CRON")
                
                with ApiClient(line_config) as client:
                    for uid in list(subscribers):
                        try: 
                            MessagingApi(client).push_message(
                                PushMessageRequest(to=uid, messages=[TextMessage(text=answer)])
                            )
                        except: pass
            # พัก 60 วินาที ป้องกันการส่งซ้ำในนาทีเดียวกัน
            await asyncio.sleep(60)
        else: 
            # ถ้ายังไม่ถึงเวลา ให้เช็ครอบใหม่ทุกๆ 30 วินาที
            await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    bg_task = asyncio.create_task(proactive_wellness_routine())
    yield
    bg_task.cancel()

app = FastAPI(title="Wellness Bot Production", lifespan=lifespan)

@app.post("/callback")
async def webhook(request: Request, x_line_signature: str = Header(None)):
    try:
        events = wh_parser.parse((await request.body()).decode(), x_line_signature)
        for event in events:
            if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
                uid = getattr(event.source, "user_id", None)
                if uid: 
                    # เมื่อมีคนทักมา จะเพิ่มเข้า list เพื่อรอรับแจ้งเตือนอัตโนมัติ
                    subscribers.add(uid)
                    asyncio.create_task(process_message(event.reply_token, uid, event.message.text))
        return Response("OK", 200)
    except: return Response("OK", 200)

@app.get("/view/{id}", response_class=HTMLResponse)
async def view_answer(id: str):
    data = answer_store.get(id)
    if not data: return HTMLResponse("<h2>⏳ ลิงก์หมดอายุแล้วน้าาา</h2>", 404)
    return HTMLResponse(f"<html><body style='padding:20px; font-family:sans-serif;'><pre style='white-space:pre-wrap;'>{html.escape(data['text'])}</pre></body></html>")
