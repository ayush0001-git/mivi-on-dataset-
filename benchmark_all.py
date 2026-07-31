import time
import numpy as np
from sentence_transformers import SentenceTransformer

def benchmark_model(model_name, dims):
    print(f"\nLoading {model_name}...")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device="cpu")
    print(f"Loaded in {time.perf_counter()-t0:.2f}s")
    
    query = "Find the best MBA college under 2 lakhs in Mumbai"
    
    # Warmup
    model.encode(query)
    
    latencies = []
    print(f"Benchmarking {model_name}...")
    for _ in range(50):
        t_start = time.perf_counter()
        model.encode(query)
        latencies.append((time.perf_counter() - t_start) * 1000)
    
    avg = np.mean(latencies)
    
    # Calculate pgvector RAM for 35k colleges
    ram_mb = (38700 * dims * 4) / (1024 * 1024)
    
    print(f"--- {model_name} Stats ---")
    print(f"Avg Latency: {avg:.2f} ms")
    print(f"Max QPS per core: {1000/avg:.2f}")
    print(f"Vector Index RAM (38.7k): {ram_mb:.2f} MB")

benchmark_model("intfloat/multilingual-e5-small", 384)
benchmark_model("intfloat/multilingual-e5-base", 768)
benchmark_model("intfloat/multilingual-e5-large", 1024)
