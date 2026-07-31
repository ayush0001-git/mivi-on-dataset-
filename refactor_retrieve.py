import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/rag_core/retrieve.py', 'r', encoding='utf-8') as f:
    text = f.read()

encode_hypo = '''
def encode_hypothetical(question: str):
    \"\"\"Generate a fake answer, embed it, use for retrieval.\"\"\"
    from .llm import chat
    from . import config
    
    try:
        prompt = f"""Write a short, factual answer to this college admissions question
(2-3 sentences, no preamble, no formatting, just the answer text):
{question}"""
        fake_answer = chat(
            [{"role": "user", "content": prompt}],
            model=config.FAST_MODEL,
            temperature=0.0,
            max_tokens=200
        )
        if not fake_answer:
            return None
        return encode_query(fake_answer)  # use query prefix on the answer
    except Exception:
        return None
'''

search_start = text.find('def search(question: str, filters: dict | None = None, top_k: int = 12, query_vec=None) -> list[dict]:')
if search_start != -1:
    text = text.replace(
        'def search(question: str, filters: dict | None = None, top_k: int = 12, query_vec=None) -> list[dict]:',
        'def search(question: str, filters: dict | None = None, top_k: int = 12, query_vec=None, use_hyde: bool = False) -> list[dict]:'
    )
    # now replace the query_vec = ... logic
    old_query_logic = '''    if query_vec is None:
        query_vec = encode_query(question)'''
    new_query_logic = '''    if use_hyde and query_vec is None:
        query_vec = encode_hypothetical(question) or encode_query(question)
    elif query_vec is None:
        query_vec = encode_query(question)'''
    text = text.replace(old_query_logic, new_query_logic)
    
    # insert encode_hypothetical above search
    search_start = text.find('def search(')
    text = text[:search_start] + encode_hypo + '\n\n' + text[search_start:]
else:
    print('Failed to find search function')
    sys.exit(1)

with open('E:/mivi on dataset/rag_core/retrieve.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated retrieve.py with HyDE')
