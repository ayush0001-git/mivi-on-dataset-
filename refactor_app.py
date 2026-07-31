import sys, re
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

streaming_code = '''
from fastapi.responses import StreamingResponse
import asyncio
import json

async def _answer_streaming(question, history, profile):
    \"\"\"Async wrapper around chat() that yields tokens.\"\"\"
    from rag_core.pipeline import _prepare_generation
    from rag_core import config
    from openai import AsyncOpenAI
    
    prep = _prepare_generation(question, history, profile)
    if "fast_return" in prep:
        yield prep["fast_return"].get("answer", "")
        return
        
    messages = prep["messages"]
    
    client = AsyncOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        timeout=config.LLM_TIMEOUT
    )
    
    stream = await client.chat.completions.create(
        model=config.GEN_MODEL,
        messages=messages,
        stream=True,
        temperature=0.0,
        max_tokens=1400
    )
    
    buffer = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            buffer += chunk.choices[0].delta.content
            # Buffer until we have a complete sentence or 50 chars
            if len(buffer) >= 50 or buffer.endswith(('.', '!', '?', '\\n')):
                yield buffer
                buffer = ""
    if buffer:
        yield buffer

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    \"\"\"Server-Sent Events streaming endpoint.\"\"\"
    _enforce_ip_limit(request)
    await _require_corpus()
    message = req.message.strip()
    if not message:
        raise HTTPException(422, "message must not be empty")
    
    snap = service.sessions.snapshot(req.session_id)
    profile = service.merge_profile(snap.profile, req.profile)
    
    async def event_generator():
        try:
            # Stage 1: token stream from LLM
            async for token_chunk in _answer_streaming(message, snap.history, profile):
                yield f"data: {json.dumps({'type': 'token', 'content': token_chunk})}\\n\\n"
            
            # Stage 2: final metadata
            yield f"data: {json.dumps({'type': 'done', 'session_id': snap.session_id})}\\n\\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'content': 'timeout'})}\\n\\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\\n\\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
'''

# insert streaming_code right before @app.get("/")
idx = text.find('@app.get("/")')
if idx == -1:
    idx = text.find('if __name__ == "__main__":')

final_text = text[:idx] + streaming_code + '\n\n' + text[idx:]

with open('E:/mivi on dataset/app.py', 'w', encoding='utf-8') as f:
    f.write(final_text)

print('Successfully added streaming to app.py')
