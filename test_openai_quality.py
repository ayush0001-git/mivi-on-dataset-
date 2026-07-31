import os
import numpy as np
from openai import OpenAI
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

# The key you provided (whitespace stripped)
api_key = "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".strip()

# Initialize OpenAI client directly connected to OpenAI (not the fallback proxy)
client = OpenAI(api_key=api_key)

def get_openai_embeddings(texts):
    embeddings = []
    for text in texts:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(np.array(resp.data[0].embedding))
    return embeddings

def get_openai_query_embedding(text):
    return get_openai_embeddings([text])[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n--- STARTING EMBEDDING QUALITY BENCHMARK ---\n")

print("[1/2] Loading intfloat/multilingual-e5-small (Your Current Model)...")
model_small = SentenceTransformer("intfloat/multilingual-e5-small")

print("[2/2] Calling OpenAI text-embedding-3-small API...")

print("\nEmbedding the 8 sample colleges (This uses less than $0.0001 in credits)...")
small_db = model_small.encode(COLLEGES)
try:
    openai_db = get_openai_embeddings(COLLEGES)
except Exception as e:
    print("\n[!] OPENAI API KEY FAILED:", getattr(e, 'message', str(e)))
    exit(1)

print("\n=======================================================")
print("--- RANKING RESULTS ---")
print("=======================================================\n")

for q in QUERIES:
    print(f"--- QUERY: \"{q}\" ---")
    
    q_small = model_small.encode(q)
    q_openai = get_openai_query_embedding(q)
    
    scores_small = [(cosine_similarity(q_small, db_vec), COLLEGES[i]) for i, db_vec in enumerate(small_db)]
    scores_openai = [(cosine_similarity(q_openai, db_vec), COLLEGES[i]) for i, db_vec in enumerate(openai_db)]
    
    scores_small.sort(key=lambda x: x[0], reverse=True)
    scores_openai.sort(key=lambda x: x[0], reverse=True)
    
    print("\n  [Rank 1 & 2] e5-small (Current Local):")
    for score, text in scores_small[:2]:
        print(f"     [{score:.3f}] {text[:60]}...")
        
    print("\n  [Rank 1 & 2] OpenAI text-embedding-3-small (API Pivot):")
    for score, text in scores_openai[:2]:
        print(f"     [{score:.3f}] {text[:60]}...")
        
    print("-" * 50)
