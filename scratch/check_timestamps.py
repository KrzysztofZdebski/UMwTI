import os
import json
import datetime

cids = {
    'Transcriber 821-840': 'fca9d714-b4f9-4873-93bb-7a74ce04d287',
    'Transcriber 841-860': 'd8905e14-77bf-46cc-a55d-16efed556ebd',
    'Batch 1': '2322df4a-e711-4cfd-ad7f-2f9139f948da',
    'Batch 2': '8747c47a-3512-4e3c-97ac-704e18d2aab3',
    'Batch 3': '43934ac8-a102-42b3-a429-d4f46d8a4068',
    'Batch 4': 'e6f32a5c-2900-4cb5-a3aa-8c83cc2a179b',
    'Batch 5': '95a8f048-a943-4b66-9be1-06a74d7bd51a',
    'Batch 6': '4c1da2f9-a38f-44bf-8aa7-d40884196957',
    'Batch 7': 'af307323-675c-4b5c-a1ed-2369e5fcfd29',
    'Batch 8': '415aa915-24cb-40ea-b928-509e300635f5',
    'Batch 9': '0808bf8c-8c72-4a68-b6e9-96a5b163b70d',
    'Batch 10': '721cec13-4d21-498e-baaf-0d3fec754cba'
}

for name, cid in cids.items():
    log_path = f'C:/Users/KMZde/.gemini/antigravity/brain/{cid}/.system_generated/logs/transcript.jsonl'
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f]
        if lines:
            last = lines[-1]
            print(f"{name}: last_step={last.get('step_index')}, created_at={last.get('created_at')}")
        else:
            print(f"{name}: empty log")
    else:
        print(f"{name}: missing log")
