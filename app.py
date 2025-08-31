from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
import os

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client with your API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def home():
    return {"message": "Edugram CBSE Chatbot API is running!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_message = body.get("message", "")

    # Call OpenAI API (GPT model)
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # lightweight & fast
        messages=[
            {"role": "system", "content": "You are a helpful AI tutor for underprivileged children. Use NCERT CBSE textbooks as your base knowledge. Explain concepts simply and clearly."},
            {"role": "user", "content": user_message}
        ],
        max_tokens=300
    )

    answer = response.choices[0].message.content
    return JSONResponse({"reply": answer})
