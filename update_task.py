import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('C:/Users/ayush/.gemini/antigravity-ide/brain/a069de7b-f9c5-4ccf-8430-13c1171a0ee8/task.md', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('- [ ] Phase 1: Streaming + Async + Redis', '- [x] Phase 1: Streaming + Async + Redis')
text = text.replace('- [ ] Phase 2: Better Embeddings + Reranking', '- [x] Phase 2: Better Embeddings + Reranking')
text = text.replace('- [ ] Phase 3: Semantic Cache + Async', '- [x] Phase 3: Semantic Cache + Async')

with open('C:/Users/ayush/.gemini/antigravity-ide/brain/a069de7b-f9c5-4ccf-8430-13c1171a0ee8/task.md', 'w', encoding='utf-8') as f:
    f.write(text)
