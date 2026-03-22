from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import base64
import os

app = FastAPI()

# Enable CORS so your Flutter app can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# We will set this variable in the Vercel Dashboard later
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def health_check():
    return {"status": "CareConnect AI is online"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Act as a medical assistant. Analyze this prescription/report. List: 1. Medicines found 2. Dosages 3. Brief summary. Disclaimer: For project demo only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
        )
        return {"analysis": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}