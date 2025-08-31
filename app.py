from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import chromadb
import os

app = Flask(__name__)
CORS(app)  # Allow requests from Wix frontend

# 🔑 OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🟢 Connect to Chroma Cloud
client = chromadb.CloudClient(
    api_key='ck-5xqbfLgmB536Cn2b8yyMccFHMhUKTdmMb7pzQMDzAj9e',
    tenant='cae1c14f-4bd7-4976-9c2b-8c3a3d4f53cd',
    database='edugram_ncert'
)

# Get or create your collection
collection = client.get_or_create_collection("edugram_ncert")

# ✅ Add sample NCERT Grade 7 content if empty
if len(collection.get()["ids"]) == 0:
    sample_chapters = [
        "Chapter 1: Nutrition in Plants: Plants make their own food via photosynthesis. They absorb water and minerals from soil, carbon dioxide from air, and use sunlight to make glucose and oxygen."
    ]
    for i, text in enumerate(sample_chapters):
        collection.add(
            documents=[text],
            metadatas=[{"chapter": i+1}],
            ids=[f"doc_{i+1}"]
        )

@app.route("/")
def index():
    return "Edugram AI Backend is running!"

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_input = request.json.get("question", "")
        if not user_input:
            return jsonify({"answer": "Please provide a question."})

        # 1️⃣ Query Chroma for relevant NCERT content
        results = collection.query(query_texts=[user_input], n_results=3)
        context_docs = results["documents"][0]
        context = "\n".join(context_docs) if context_docs else "No relevant NCERT content found."

        # 2️⃣ Send context + question to OpenAI GPT
        prompt = f"""
        You are Edugram AI, a helpful tutor for students (Grade 7-12).
        Use the following NCERT material to answer the question.

        Context:
        {context}

        Question: {user_input}

        Answer in simple, clear language as if explaining to a student.
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
