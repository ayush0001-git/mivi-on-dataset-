import time
import numpy as np
from rag_core.retrieve import encode_query

query = "Find the best MBA college under 2 lakhs in Mumbai"

latencies = []

# Warm up the model first so it's loaded in memory and we don't count load time
print("Warming up model...")
encode_query(query)

print("Starting benchmark...")
for i in range(100):
    embedding_start = time.perf_counter()
    qv = encode_query(query)
    embedding_ms = (time.perf_counter() - embedding_start) * 1000
    latencies.append(embedding_ms)
    print(f"Embedding Time: {embedding_ms:.2f} ms")

print("\n--- Results ---")
print(f"Average embedding time: {np.mean(latencies):.2f} ms")
print(f"P95 embedding time: {np.percentile(latencies, 95):.2f} ms")
print(f"Maximum embedding time: {np.max(latencies):.2f} ms")
