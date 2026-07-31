import sys, re
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_logic = '''
    sem_cache_result = service.semantic_cache.get(message, profile) if not snap.history else None
    if sem_cache_result:
        result = sem_cache_result
        cached = True
        request.state.cache_hit = True
    else:
        cached = False
        request.state.cache_hit = False
        try:
'''

old_logic = '''    cache_key = service.answer_cache_key(message, profile) if not snap.history else None
    result = service.answer_cache.get(cache_key) if cache_key else None
    cached = result is not None
    request.state.cache_hit = cached

    if not cached:
        try:'''

text = text.replace(old_logic, new_logic)

# find where it puts to cache
put_old = '''        if cache_key and result.get("answered"):
            service.answer_cache.put(cache_key, service.project_answer(result))'''
put_new = '''        if not snap.history and result.get("answered"):
            service.semantic_cache.put(message, profile, service.project_answer(result))'''
text = text.replace(put_old, put_new)

with open('E:/mivi on dataset/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated app.py with semantic cache')
