import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/rag_core/pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_search = '''
def _search(question, filters, top_k, query_vec=None):
    \"\"\"retrieve.search with the store's failure surfaced, not swallowed.\"\"\"
    hits = retrieve.search(question, filters=filters or None, top_k=top_k,
                           query_vec=query_vec)
    from . import config
    from . import rerank
    if hits and config.RERANK_ENABLED:
        # Only rerank if confidence is low
        top_rerank_score = hits[0].get("score", 0)
        if top_rerank_score < config.RERANK_CONFIDENCE_THRESHOLD:
            hits = rerank.rerank(question, hits, top_k=top_k)
    return hits
'''

start_idx = text.find('def _search(question, filters, top_k, query_vec=None):')
if start_idx != -1:
    end_idx = text.find('def _count(', start_idx)
    text = text[:start_idx] + new_search + '\n\n' + text[end_idx:]

with open('E:/mivi on dataset/rag_core/pipeline.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated _search in pipeline.py')
