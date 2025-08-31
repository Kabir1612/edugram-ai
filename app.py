from flask import Flask, render_template, request, jsonify
import openai
import chromadb
import os

app = Flask(__name__)

# 🔑 OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🟢 Connect to Chroma
client = chromadb.PersistentClient(path="./chroma_db")  # database stored locally
collection = client.get_or_create_collection("edugram_ncert_7_12")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json["question"]

    # 1️⃣ Search Chroma for relevant NCERT content
    results = collection.query(query_texts=[user_input], n_results=3)
    context = "\n".join([doc for doc in results["documents"][0]])

    # 2️⃣ Send context + question to GPT
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
