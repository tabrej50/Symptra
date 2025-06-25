
import torch
import faiss
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

df = pd.read_csv("sample_symptra.csv")
qa_texts = [f"question: {q}\nanswer: {a}" for q, a in zip(df["question"], df["answer"])]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(qa_texts, convert_to_tensor=False)

dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

def symptra_chat(query):
    query_embed = embedder.encode([query])
    D, I = index.search(query_embed, k=2)
    context = "\n\n".join([qa_texts[i] for i in I[0]])
    prompt = f"You are Symptra, an AI doctor trained on clinical guidelines.\n\nContext:\n{context}\n\nUser: {query}\nSymptra:"
    result = generator(prompt)[0]["generated_text"]
    return result[len(prompt):].strip()
