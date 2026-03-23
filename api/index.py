from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import base64
import os

app = FastAPI()

# Enable CORS for Flutter communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HARDCODED API KEY FOR DEMO ---
# PASTE YOUR KEY FROM GROQ CONSOLE HERE
GROQ_API_KEY = "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" 
client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def health_check():
    return {"status": "CareConnect AI is online"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # 1. Clean Base64 (Extreme cleanup)
        base64_image = base64.b64encode(contents).decode('utf-8')
        clean_base64 = "".join(base64_image.split())
        
        content_type = file.content_type or "image/jpeg"

        # 2. Call Groq with the EXACT model string
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview", # Verify this name in Groq docs
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List the medicines and dosages in this image."},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:{content_type};base64,{clean_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        # RETURN THE FULL ERROR so we can see exactly what Groq hates
        return {"analysis": f"AI Error Detail: {str(e)}"}