import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/rag_core/config.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_config = '''
EMBED_MODEL_OPTIONS = {
    "e5-small": "intfloat/multilingual-e5-small",       # current, 384 dim
    "e5-base": "intfloat/multilingual-e5-base",          # 768 dim, better quality
    "e5-large": "intfloat/multilingual-e5-large",        # 1024 dim, best e5
    "bge-m3": "BAAI/bge-m3",                            # 1024 dim, best multilingual
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 768 dim
    "cohere": "embed-multilingual-v3",                   # API, 1024 dim
}

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

RERANK_ENABLED = os.getenv("MIVI_RERANK", "1") == "1"
RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("MIVI_RERANK_THRESHOLD", "0.01"))
'''

# We also added RERANK_ENABLED here for phase 2.2

with open('E:/mivi on dataset/rag_core/config.py', 'w', encoding='utf-8') as f:
    f.write(text + '\n' + new_config + '\n')

print('Updated config.py')
