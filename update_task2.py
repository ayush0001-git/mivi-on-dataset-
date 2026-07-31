import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('C:/Users/ayush/.gemini/antigravity-ide/brain/a069de7b-f9c5-4ccf-8430-13c1171a0ee8/task.md', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('- [ ] Phase 4: RAG Triad Metrics', '- [x] Phase 4: RAG Triad Metrics')
text = text.replace('- [ ] Phase 5: Data: NIRF/NAAC/Cutoffs', '- [x] Phase 5: Data: NIRF/NAAC/Cutoffs')
text = text.replace('- [ ] Phase 6: UI: Streaming + Multilingual', '- [x] Phase 6: UI: Streaming + Multilingual')

with open('C:/Users/ayush/.gemini/antigravity-ide/brain/a069de7b-f9c5-4ccf-8430-13c1171a0ee8/task.md', 'w', encoding='utf-8') as f:
    f.write(text)
