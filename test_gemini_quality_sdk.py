import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

COLLEGES = [
    "Pune Institute of Business Management (PIBM). Course: MBA. Fees: 1.5 Lakhs per year. Excellent placements in marketing.",
    "Symbiosis Institute of Business Management (SIBM) Pune. Course: MBA. Fees: 12 Lakhs per year. Top tier college.",
    "Delhi Technological University (DTU). Course: BTech Computer Science. Fees: 2.5 Lakhs per year. Government college.",
    "SRM University Chennai. Course: BTech Computer Science. Fees: 4.5 Lakhs per year. High tech campus.",
    "G.H. Raisoni College of Engineering, Nagpur. Course: BTech. Fees: 1.2 Lakhs. Affordable engineering in Maharashtra.",
    "Indian Institute of Technology (IIT) Bombay. Course: BTech. Fees: 2 Lakhs per year. The best in India.",
    "Lovely Professional University (LPU) Punjab. Course: BTech. Fees: 1.5 Lakhs per year. Massive campus.",
    "Amity University Noida. Course: BBA. Fees: 3 Lakhs per year. Great infrastructure."
]

QUERIES = [
    "Top engineering college in Maharashtra",
    "MBA college in Pune under 2 lakhs",
    "saste btech colleges pune ya maharashtra mein"
]

# Get Gemini API key from .env manually for this test
env_key = None
with open(".env", "r") as f:
    for line in f:
        if line.startswith("FALLBACK_API_KEY="):
            env_key = line.strip().split("=")[1]
            break

genai.configure(api_key=env_key)

def get_gemini_embeddings(texts):
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(np.array(result['embedding']))
    return embeddings

def get_gemini_query_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result['embedding'])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n--- STARTING EMBEDDING QUALITY BENCHMARK ---\n")

print("[1/2] Loading intfloat/multilingual-e5-small (Your Current Model)...")
model_small = SentenceTransformer("intfloat/multilingual-e5-small")

print("[2/2] Calling Google Gemini API (Cloud Option)...")

print("\nEmbedding the 8 sample colleges...")
small_db = model_small.encode(COLLEGES)
gemini_db = get_gemini_embeddings(COLLEGES)

print("\n=======================================================")
print("--- RANKING RESULTS ---")
print("=======================================================\n")

for q in QUERIES:
    print(f"--- QUERY: \"{q}\" ---")
    
    q_small = model_small.encode(q)
    q_gemini = get_gemini_query_embedding(q)
    
    scores_small = [(cosine_similarity(q_small, db_vec), COLLEGES[i]) for i, db_vec in enumerate(small_db)]
    scores_gemini = [(cosine_similarity(q_gemini, db_vec), COLLEGES[i]) for i, db_vec in enumerate(gemini_db)]
    
    scores_small.sort(key=lambda x: x[0], reverse=True)
    scores_gemini.sort(key=lambda x: x[0], reverse=True)
    
    print("\n  [Rank 1 & 2] e5-small (Current Local):")
    for score, text in scores_small[:2]:
        print(f"     [{score:.3f}] {text[:60]}...")
        
    print("\n  [Rank 1 & 2] Gemini text-embedding-004 (API Pivot):")
    for score, text in scores_gemini[:2]:
        print(f"     [{score:.3f}] {text[:60]}...")
        
    print("-" * 50)
