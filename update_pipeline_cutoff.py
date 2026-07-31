import sys
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/rag_core/pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_func = '''
def lookup_cutoff(college_id: str, exam: str, year: int, category: str = "GENERAL") -> dict | None:
    from .store.backend import get_backend
    rows = get_backend().q("""
        SELECT year, round, opening_rank, closing_rank
        FROM cutoff
        WHERE college_id = %s AND exam = %s AND category = %s AND year = %s
        ORDER BY round DESC LIMIT 5
    """, [college_id, exam, category, year])
    return [dict(r) for r in rows] if rows else None
'''

# append new_func to pipeline.py
with open('E:/mivi on dataset/rag_core/pipeline.py', 'a', encoding='utf-8') as f:
    f.write('\n' + new_func + '\n')

print('Added lookup_cutoff to pipeline.py')
