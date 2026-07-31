import sys, re
sys.stdout.reconfigure(encoding='utf-8')
with open('E:/mivi on dataset/rag_core/pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

# We want to replace def answer_question(question, history=None, known_profile=None):
# with a new function def _prepare_generation(question, history=None, known_profile=None):
# which returns everything needed.
# And then redefine def answer_question(...) to use it.

start_str = 'def answer_question(question, history=None, known_profile=None):'
start_idx = text.find(start_str)

if start_idx == -1:
    print('Failed to find answer_question')
    sys.exit(1)

# Find where the generation loop starts:
loop_str = '    result, verified = None, False\n    for attempt in range(3):'
loop_idx = text.find(loop_str, start_idx)

if loop_idx == -1:
    print('Failed to find loop')
    sys.exit(1)

header_text = text[start_idx:loop_idx]

# Change def answer_question to def _prepare_generation
header_text = header_text.replace(
    'def answer_question(question, history=None, known_profile=None):',
    'def _prepare_generation(question, history=None, known_profile=None):'
)

# Replace returns of fast smalltalk/cache hits to return special dicts
header_text = re.sub(
    r'return result\n',
    r'return {"fast_return": result}\n',
    header_text
)
header_text = re.sub(
    r'return fast\n',
    r'return {"fast_return": fast}\n',
    header_text
)

# Return the prepared arguments for generation
return_stmt = '''    return {
        "messages": messages,
        "extra_ok": extra_ok,
        "calls": calls,
        "context_ids": context_ids,
        "total": total,
        "verified": verified,
        "route": route,
        "merged_profile": merged_profile,
        "cache_key": cache_key,
        "t_start": t_start
    }
'''

new_answer_question = '''
def answer_question(question, history=None, known_profile=None):
    prep = _prepare_generation(question, history, known_profile)
    if "fast_return" in prep:
        return prep["fast_return"]
    
    messages = prep["messages"]
    extra_ok = prep["extra_ok"]
    calls = prep["calls"]
    context_ids = prep["context_ids"]
    total = prep["total"]
    verified = prep["verified"]
    route = prep["route"]
    merged_profile = prep["merged_profile"]
    cache_key = prep["cache_key"]
    t_start = prep["t_start"]
    
    result, verified = None, False
    for attempt in range(3):
'''

text_after_loop = text[loop_idx + len(loop_str):]

# Wait, or attempt in range(3): is at loop_idx.
# I just need to stitch this all together.
final_text = text[:start_idx] + header_text + return_stmt + new_answer_question + text_after_loop

with open('E:/mivi on dataset/rag_core/pipeline.py', 'w', encoding='utf-8') as f:
    f.write(final_text)

print('Successfully refactored pipeline.py')
