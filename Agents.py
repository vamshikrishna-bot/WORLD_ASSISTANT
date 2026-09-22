from dotenv import load_dotenv

load_dotenv()

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient


@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "ERROR: OpenWeather API key is missing."

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if str(data.get("cod")) != "200":
            return f"ERROR: {data.get('message', 'Could not fetch weather')}"
        return (
            "CURRENT_WEATHER\n"
            f"City: {city.title()}\n"
            f"Temperature: {data['main']['temp']}°C\n"
            f"Feels Like: {data['main']['feels_like']}°C\n"
            f"Condition: {data['weather'][0]['description'].title()}\n"
            f"Humidity: {data['main']['humidity']}%\n"
            f"Wind Speed: {data['wind']['speed']} m/s\n"
            f"Pressure: {data['main']['pressure']} hPa"
        )
    except requests.exceptions.Timeout:
        return "ERROR: Weather service request timed out."
    except Exception as error:
        return f"ERROR: {error}"


tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""
    if tavily_client is None:
        return "ERROR: Tavily API key is missing."
    try:
        response = tavily_client.search(
            query=f"latest news in {city}", search_depth="basic", max_results=5
        )
        results = response.get("results", [])
        if not results:
            return f"NO_NEWS\nNo latest news was found for {city}."
        articles = []
        for result in results:
            articles.append(
                f"TITLE: {result.get('title', 'No title')}\n"
                f"DESCRIPTION: {result.get('content', 'No description available.')[:250]}\n"
                f"URL: {result.get('url', '')}"
            )
        return "CITY_NEWS\n" f"City: {city.title()}\n\n" + "\n\n".join(articles)
    except Exception as error:
        return f"ERROR: {error}"


if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is missing from your .env file.")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="""
You are a professional City Assistant.
You help users with current weather and latest city news.
Use get_weather for current weather and get_news for latest news.
Do not invent weather or news. Give concise, professional answers.
Do not show internal tool information, Python dictionaries, tool signatures,
or internal metadata. For weather, mention city, temperature, condition,
humidity, wind, and feels-like temperature. For news, summarize the results clearly.
""",
)


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>City Assistant</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--ink:#17201d;--muted:#68736e;--paper:#f4f1e9;--card:#ffffffb8;--line:#17201d1f;--lime:#c9f06a;--teal:#186d62;--coral:#ed795f}*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--ink);font-family:'Space Grotesk',sans-serif;background:var(--paper);overflow-x:hidden}body:before{content:'';position:fixed;inset:0;z-index:-2;background:radial-gradient(circle at 80% 8%,#c9f06a85,transparent 28%),radial-gradient(circle at 4% 92%,#ed795f45,transparent 25%),linear-gradient(125deg,#f7f4ed,#e5eee5 55%,#f8eadf)}body:after{content:'';position:fixed;inset:0;z-index:-1;opacity:.3;background-image:linear-gradient(#17201d0f 1px,transparent 1px),linear-gradient(90deg,#17201d0f 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(#000,transparent 80%)}.shell{display:grid;grid-template-columns:280px minmax(0,1fr);min-height:100vh;max-width:1500px;margin:auto}aside{padding:30px 24px;border-right:1px solid var(--line);background:#ffffff42;backdrop-filter:blur(16px)}.mark{display:flex;align-items:center;gap:11px;font-weight:700;letter-spacing:-.04em;font-size:20px}.mark-dot{width:34px;height:34px;display:grid;place-items:center;border-radius:11px;background:var(--ink);color:var(--lime);box-shadow:0 7px 18px #17201d33}.eyebrow{margin:72px 0 12px;font:500 10px 'DM Mono',monospace;letter-spacing:.16em;color:var(--muted);text-transform:uppercase}aside h2{font-size:20px;line-height:1.15;margin:0 0 12px;letter-spacing:-.05em}aside p{color:var(--muted);font-size:13px;line-height:1.6;margin:0}.tool-list{display:grid;gap:10px;margin-top:28px}.tool-chip{display:flex;gap:12px;align-items:center;padding:12px;border:1px solid var(--line);border-radius:14px;background:#ffffff61;font-size:13px}.tool-chip span{display:block;margin-top:2px;color:var(--muted);font-size:11px}.icon{display:grid;place-items:center;flex:none;width:32px;height:32px;border-radius:10px;background:var(--lime)}.tip{margin-top:38px;padding:15px;border-radius:16px;background:var(--ink);color:#fff;font-size:12px;line-height:1.55}.tip strong{color:var(--lime);display:block;margin-bottom:5px}main{min-width:0;padding:44px clamp(24px,6vw,88px) 38px}.topline{display:flex;justify-content:space-between;align-items:center;gap:20px;animation:rise .7s ease both}.status{display:flex;align-items:center;gap:8px;font:500 11px 'DM Mono',monospace;color:var(--muted)}.pulse{width:8px;height:8px;border-radius:50%;background:#4cae70;box-shadow:0 0 0 #4cae70;animation:pulse 2s infinite}.hero{max-width:760px;margin:clamp(58px,11vh,124px) auto 52px;text-align:center;animation:rise .8s .1s ease both}.hero h1{margin:0;font-size:clamp(44px,7vw,88px);line-height:.94;letter-spacing:-.085em}.hero h1 em{color:var(--teal);font-style:normal}.hero p{max-width:470px;margin:22px auto 0;color:var(--muted);line-height:1.6;font-size:15px}#messages{max-width:820px;margin:auto;display:grid;gap:16px}.message{display:flex;gap:12px;animation:rise .45s ease both}.message.user{justify-content:flex-end}.bubble{max-width:min(680px,85%);padding:15px 18px;border:1px solid var(--line);border-radius:18px 18px 18px 5px;background:var(--card);backdrop-filter:blur(12px);line-height:1.6;font-size:14px;box-shadow:0 10px 30px #17201d0f;white-space:pre-wrap}.user .bubble{background:var(--ink);color:#fff;border:0;border-radius:18px 18px 5px 18px}.assistant-tag{font:500 10px 'DM Mono',monospace;color:var(--teal);margin:0 0 5px 3px;letter-spacing:.08em}.composer{max-width:820px;margin:30px auto 0}form{display:flex;align-items:center;gap:10px;padding:8px 8px 8px 18px;border:1px solid #17201d33;border-radius:18px;background:#ffffffbf;box-shadow:0 20px 45px #17201d1a;transition:.25s}form:focus-within{border-color:var(--teal);box-shadow:0 0 0 4px #186d621f,0 20px 45px #17201d1a}input{width:100%;border:0;outline:0;color:var(--ink);background:transparent;font:14px 'Space Grotesk',sans-serif}button{border:0;cursor:pointer}.send{width:42px;height:42px;flex:none;border-radius:13px;background:var(--ink);color:var(--lime);font-size:18px;transition:.2s}.send:hover{transform:translateY(-2px) rotate(-4deg);background:var(--teal)}.suggestions{display:flex;justify-content:center;flex-wrap:wrap;gap:8px;margin-top:14px}.suggestions button{padding:8px 12px;border:1px solid var(--line);border-radius:999px;color:var(--muted);background:#ffffff66;font:12px 'Space Grotesk',sans-serif;transition:.2s}.suggestions button:hover{color:var(--ink);background:var(--lime);transform:translateY(-2px)}.typing{display:flex;gap:4px;align-items:center;padding:5px 0}.typing i{width:5px;height:5px;border-radius:50%;background:var(--teal);animation:bounce 1s infinite}.typing i:nth-child(2){animation-delay:.15s}.typing i:nth-child(3){animation-delay:.3s}.footer{text-align:center;margin-top:42px;color:var(--muted);font:10px 'DM Mono',monospace;letter-spacing:.08em}@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}@keyframes pulse{70%{box-shadow:0 0 0 7px #4cae7000}100%{box-shadow:0 0 0 #4cae7000}}@keyframes bounce{0%,100%{transform:translateY(0);opacity:.4}50%{transform:translateY(-4px);opacity:1}}@media(max-width:760px){.shell{display:block}aside{border-right:0;border-bottom:1px solid var(--line);padding:18px 20px}.eyebrow,.tool-list,.tip,aside p{display:none}main{padding:24px 18px 28px}.hero{margin:70px auto 40px}.hero h1{font-size:clamp(48px,15vw,74px)} }
</style></head>
<body><div class="shell"><aside><div class="mark"><div class="mark-dot">✦</div> city assistant</div><div class="eyebrow">Your local briefing desk</div><h2>Know your city<br>before you step out.</h2><p>Ask for live conditions or a quick scan of what is making the headlines.</p><div class="tool-list"><div class="tool-chip"><div class="icon">☼</div><div><b>Live weather</b><span>Temperature, wind & more</span></div></div><div class="tool-chip"><div class="icon">◈</div><div><b>City news</b><span>Fresh local headlines</span></div></div></div><div class="tip"><strong>Try a prompt</strong>“What is the weather in Hyderabad?”</div></aside><main><div class="topline"><span class="status"><i class="pulse"></i> SYSTEMS ONLINE</span><span class="status">GEMINI / TAVILY / OPENWEATHER</span></div><section class="hero"><h1>Your city,<br><em>in the know.</em></h1><p>A calm, quick way to check the weather and catch up on the latest local stories.</p></section><section id="messages" aria-live="polite"></section><div class="composer"><form id="chat-form"><input id="prompt" autocomplete="off" placeholder="Ask about a city..." aria-label="Ask about a city"><button class="send" aria-label="Send message">↗</button></form><div class="suggestions"><button type="button">Weather in Hyderabad</button><button type="button">Latest news in Mumbai</button><button type="button">Is it raining in Delhi?</button></div></div><div class="footer">A SMALL WINDOW INTO THE WORLD AROUND YOU</div></main></div>
<script>const messages=document.querySelector('#messages'),form=document.querySelector('#chat-form'),prompt=document.querySelector('#prompt');function addMessage(text,role,loading=false){const row=document.createElement('div');row.className=`message ${role}`;const bubble=document.createElement('div');bubble.className='bubble';if(loading)bubble.innerHTML='<div class="assistant-tag">CITY ASSISTANT</div><div class="typing"><i></i><i></i><i></i></div>';else{if(role==='assistant'){const tag=document.createElement('div');tag.className='assistant-tag';tag.textContent='CITY ASSISTANT';bubble.append(tag)}const body=document.createElement('div');body.textContent=text;bubble.append(body)}row.append(bubble);messages.append(row);row.scrollIntoView({behavior:'smooth',block:'nearest'});return row}document.querySelectorAll('.suggestions button').forEach(button=>button.onclick=()=>{prompt.value=button.textContent;form.requestSubmit()});form.onsubmit=async event=>{event.preventDefault();const text=prompt.value.trim();if(!text)return;prompt.value='';addMessage(text,'user');const loading=addMessage('','assistant',true);try{const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});const data=await response.json();loading.remove();addMessage(data.response||'I could not find an answer just now.','assistant')}catch(error){loading.remove();addMessage('The assistant is temporarily unavailable. Please check the server and try again.','assistant')}};</script></body></html>'''


def clean_response(message):
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
            if isinstance(item, str)
            or (isinstance(item, dict) and item.get("type") == "text")
        ).strip()
    return str(content)


class CityAssistantHandler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
            user_input = str(request.get("message", "")).strip()
            if not user_input:
                self.send_json({"error": "Message is required."}, 400)
                return
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            self.send_json({"response": clean_response(result["messages"][-1])})
        except Exception as error:
            self.send_json({"error": str(error)}, 500)

    def log_message(self, format, *args):
        print(f"[city-assistant] {args[0]}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8501"))
    server = ThreadingHTTPServer(("127.0.0.1", port), CityAssistantHandler)
    print(f"City Assistant running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping City Assistant...")
        server.server_close()
