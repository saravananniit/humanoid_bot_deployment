import json
import tempfile
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq

app = FastAPI(title="Humanoid AI Bot")
app.mount("/static", StaticFiles(directory="static"), name="static")

SYSTEM_PROMPT = (
    "You are a friendly humanoid assistant talking face to face with a person. "
    "Reply in plain spoken sentences: no markdown, no lists, no emojis. "
    "Keep answers under 60 words unless asked for detail. "
    "If the person's message is not a question or a request, do not elaborate: "
    "reply with one short sentence inviting them to ask a question."
)


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/api/chat")
async def chat(
    api_key: str = Form(...),          # passed at runtime from the browser
    model: str = Form(...),            # chat model, e.g. llama-3.3-70b-versatile
    stt_model: str = Form("whisper-large-v3-turbo"),
    text: str = Form(""),
    history: str = Form("[]"),
    audio: UploadFile | None = File(None),
):
    if not api_key.strip():
        raise HTTPException(400, "Groq API key is required.")

    client = Groq(api_key=api_key.strip())
    transcript = text.strip()

    try:
        # 1) Speech -> text (only when audio was sent)
        if audio is not None:
            data = await audio.read()
            if len(data) < 1000:
                raise HTTPException(400, "Recording was too short. Hold the mic a bit longer.")
            suffix = ".webm" if "webm" in (audio.content_type or "") else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(data)
                path = f.name
            try:
                with open(path, "rb") as fh:
                    result = client.audio.transcriptions.create(
                        file=(os.path.basename(path), fh.read()),
                        model=stt_model,
                        response_format="text",
                    )
                transcript = (result if isinstance(result, str) else result.text).strip()
            finally:
                os.unlink(path)

        if not transcript:
            raise HTTPException(400, "I didn't catch any words. Please try again.")

        # 2) Text -> reply
        try:
            past = json.loads(history)[-12:]
        except json.JSONDecodeError:
            past = []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *past,
                    {"role": "user", "content": transcript}]
        completion = client.chat.completions.create(
            model=model.strip(), messages=messages, temperature=0.6, max_tokens=300
        )
        reply = completion.choices[0].message.content.strip()
        return {"transcript": transcript, "reply": reply}

    except HTTPException:
        raise
    except Exception as e:  # surface Groq errors (bad key, bad model) to the UI
        raise HTTPException(502, f"Groq error: {e}")
