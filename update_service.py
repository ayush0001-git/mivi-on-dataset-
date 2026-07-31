import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/rag_core/service.py', 'r', encoding='utf-8') as f:
    text = f.read()

if 'semantic_cache =' not in text:
    new_imports = '''from .semantic_cache import SemanticCache
semantic_cache = SemanticCache()
'''
    # insert after imports
    idx = text.find('class AnswerCache:')
    if idx != -1:
        text = text[:idx] + new_imports + '\n' + text[idx:]
    with open('E:/mivi on dataset/rag_core/service.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Added semantic_cache to service.py')
else:
    print('semantic_cache already in service.py')
