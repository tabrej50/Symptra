import torch
import faiss
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# ✅ Load CSV data
df = pd.read_csv("sample_symptra.csv")
qa_texts = [f"question: {q}\nanswer: {a}" for q, a in zip(df["question"], df["answer"])]

# ✅ Embedding setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(qa_texts, convert_to_tensor=False)

dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ✅ Load small model (lightweight)
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

# ✅ Core chat function
def symptra_chat(query):
    query_embed = embedder.encode([query])
    D, I = index.search(query_embed, k=2)
    context = "\n\n".join([qa_texts[i] for i in I[0]])

    prompt = f"""You are Symptra, an AI doctor trained on clinical guidelines.

📚 Context:
{context}

❓ Question: {query}

✍️ Format your answer as:
- Probable diagnosis
- Recommended tests
- Referral suggestions
- Treatment plan
- Red flag warning signs
- Home/lifestyle care
- ⚠️ Always end with: "Consult a qualified physician for confirmation."

Symptra:"""

    output = generator(prompt)[0]["generated_text"]
    return output[len(prompt):].strip()

