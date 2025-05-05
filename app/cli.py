import os
import json
from app.rag import retrieve
from openai import OpenAI
from app.config import OPENAI_API_KEY

# 1) Load your raw documents into a dict: doc_id → text
with open("data/raw_documents/metadata.json") as f:
    metadata = json.load(f)

# 2) Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def answer_question(question: str, k: int = 3):
    # 3) Retrieve top-k IDs
    ids = retrieve(question, k)
    # 4) Gather their texts
    contexts = [metadata.get(str(i), "") for i in ids if i != -1]
    # 5) Build prompt
    prompt = (
       "Use the following context to answer the question.\n\n"
       + "\n\n".join(f"Context {i}: {text}" for i, text in zip(ids, contexts))
       + f"\n\nQuestion: {question}\nAnswer:"
    )
    # 6) Call GPT-4
    resp = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    q = input("Enter your question: ")
    print("\n🔎 Retrieved contexts:", retrieve(q))
    print("\n💡 Answer:\n", answer_question(q))
