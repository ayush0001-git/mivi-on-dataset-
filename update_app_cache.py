import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace AnswerCache usage with SemanticCache
# In app.py chat() handler
# cache_key = service.answer_cache_key(message, snap.profile)
# cached = service.answers.get(cache_key)
# if cached:
#     result = cached

# I will find 'cached = service.answers.get(' and replace it
idx = text.find('cache_key = service.answer_cache_key(')
if idx != -1:
    end_idx = text.find('else:', idx)
    if end_idx != -1:
        # wait, if I can't find it exactly, I'll just write a regex
        import re
        text = re.sub(
            r'cache_key = service\.answer_cache_key[^\n]+\n\s*cached = service\.answers\.get\(cache_key\)\n\s*if cached:\n\s*result = cached\n\s*request\.state\.cache_hit = True\n\s*else:',
            r'''sem_cache_result = service.semantic_cache.get(message, profile)
    if sem_cache_result:
        result = sem_cache_result
        cached = True
        request.state.cache_hit = True
    else:''',
            text
        )
        
        # also update the put
        text = re.sub(
            r'if cache_key and result\.get\("answered"\):\n\s*service\.answers\.put\(cache_key, service\.project_answer\(result\)\)',
            r'''if result.get("answered"):
        service.semantic_cache.put(message, profile, service.project_answer(result))''',
            text
        )
        with open('E:/mivi on dataset/app.py', 'w', encoding='utf-8') as f:
            f.write(text)
        print('Updated app.py with semantic cache')
    else:
        print('Could not find else block')
else:
    print('Could not find cache_key assignment in app.py')
