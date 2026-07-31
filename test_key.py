import os
from openai import OpenAI

# Try raw
keys = [
    "sk-svcacct-uv65nhF7VBZhyf40ZZEMn_8UmlGBHLGOWUki_0Wpr5DGgS0rTchJtTDan66gOoIbXT3BlbkFJVdpROh25HVsQuus3ypkoQXlbvwfs81XD8WytKq8BDrwGrlHdl-iVh5EDNxhFg1aEGIyDHtuvEA",
    "sk-uv65nhF7VBZhyf40ZZEMn_8UmlGBHLGOWUki_0Wpr5DGgS0rTchJtTDan66gOoIbXT3BlbkFJVdpROh25HVsQuus3ypkoQXlbvwfs81XD8WytKq8BDrwGrlHdl-iVh5EDNxhFg1aEGIyDHtuvEA"
]

valid_key = None
for k in keys:
    try:
        client = OpenAI(api_key=k)
        resp = client.embeddings.create(model="text-embedding-3-small", input="test")
        print("VALID KEY:", k[:15] + "...")
        valid_key = k
        break
    except Exception as e:
        print("Failed:", k[:15], getattr(e, 'message', str(e)))

if valid_key:
    with open(".env", "a") as f:
        f.write(f"\nOPENAI_API_KEY={valid_key}\n")
    print("Successfully added to .env!")
else:
    print("ALL KEYS FAILED!")
