import os
import json

cids = {
    'Batch 1': 'e69f92b1-c600-4db8-b79b-96a30bdd1921',
    'Batch 2': '4472c5de-2c74-436b-8258-929dad76a8f8',
    'Batch 3': '654d2734-816e-4fbf-8761-a7c2df3550e0',
    'Batch 4': 'a6978a81-5e3e-4bf3-a940-7ce996db0291',
    'Batch 5': '1fde952b-a514-4067-bf1d-f932c7956b3a',
    'Batch 6': '5f84d078-0aa1-4012-9b19-c56b2822e01f',
    'Batch 7': '9fc5f8e1-3157-4ce0-9204-b296606edf45',
    'Batch 8': '7bfc1eea-4122-4c7e-a8c3-c5856133399d',
    'Batch 9': '0938c05d-fb20-4e7f-a56d-c55e1b2288c1'
}

for name, cid in sorted(cids.items()):
    log_path = f'C:/Users/KMZde/.gemini/antigravity/brain/{cid}/.system_generated/logs/transcript.jsonl'
    if not os.path.exists(log_path):
        print(f"{name}: log missing")
        continue
        
    viewed = set()
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            step = json.loads(line)
            if 'tool_calls' in step and step['tool_calls']:
                for tc in step['tool_calls']:
                    if tc.get('name') == 'view_file':
                        path = tc.get('args', {}).get('AbsolutePath', '')
                        if 'sample_1000' in path:
                            viewed.add(os.path.basename(path.replace('"', '').replace("'", '')))
                            
    print(f"{name} ({cid[:8]}): viewed {len(viewed)} images so far: {sorted(list(viewed))}")
