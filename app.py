from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ======================
# LOAD EXCEL
# ======================
url = "https://raw.githubusercontent.com/gigiel2011/chatbot-rag-CLQ/main/data.xlsx"
df = pd.read_excel(url, engine='openpyxl')

# ======================
# SIMPLE RAG (TF-IDF)
# ======================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

def search_context(question, top_k=5):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, X).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    return "\n".join(df.iloc[top_idx]["text"])

# ======================
# GROQ
# ======================
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask_groq(question, context):
    prompt = f"""
Jawab HANYA berdasarkan data ini.
Jika tiada, jawab "Tiada dalam data".

DATA:
{context}

SOALAN:
{question}
"""
    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# ======================
# API
# ======================
app = FastAPI()

class Chat(BaseModel):
    question: str

@app.post("/chat")
def chat(req: Chat):
    context = search_context(req.question)
    return {"answer": ask_groq(req.question, context)}
