import os
import requests
import numpy as np

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

env_key = None
with open(".env", "r") as f:
    for line in f:
        if line.startswith("FALLBACK_API_KEY="):
            env_key = line.strip().split("=")[1]
            break

def get_gemini_embeddings(texts):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:batchEmbedContents?key={env_key}"
    requests_data = [{"model": "models/embedding-001", "content": {"parts": [{"text": t}]}} for t in texts]
    
    resp = requests.post(url, json={"requests": requests_data})
    
    if resp.status_code != 200:
        print("API ERROR:", resp.text)
        return [np.zeros(768) for _ in texts]
        
    embeddings = resp.json().get("embeddings", [])
    if not embeddings:
        print("NO EMBEDDINGS RETURNED:", resp.text)
        return [np.zeros(768) for _ in texts]
        
    return [np.array(e["values"]) for e in embeddings]

def get_gemini_query_embedding(text):
    return get_gemini_embeddings([text])[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n--- GEMINI API QUALITY TEST ---\n")
print("Embedding the 8 sample colleges via Cloud API...")
gemini_db = get_gemini_embeddings(COLLEGES)

print("\n=======================================================")
print("--- RANKING RESULTS ---")
print("=======================================================\n")

for q in QUERIES:
    print(f"--- QUERY: \"{q}\" ---")
    q_gemini = get_gemini_query_embedding(q)
    
    scores_gemini = [(cosine_similarity(q_gemini, db_vec), COLLEGES[i]) for i, db_vec in enumerate(gemini_db)]
    scores_gemini.sort(key=lambda x: x[0], reverse=True)
    
    for score, text in scores_gemini[:2]:
        print(f"     [{score:.3f}] {text[:60]}...")
    print("-" * 50)
