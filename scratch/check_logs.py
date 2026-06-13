import os
import json

retrans_cids = {
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

for name, cid in retrans_cids.items():
    log_path = f'C:/Users/KMZde/.gemini/antigravity/brain/{cid}/.system_generated/logs/transcript.jsonl'
    if not os.path.exists(log_path):
        print(f'{name}: log missing')
        continue
    
    last_model_msg = None
    has_write = False
    write_args = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            step = json.loads(line)
            if step.get('source') == 'MODEL' and step.get('content'):
                last_model_msg = step['content']
            if 'tool_calls' in step and step['tool_calls']:
                for tc in step['tool_calls']:
                    if tc.get('name') == 'write_to_file':
                        has_write = True
                        write_args = tc.get('args')
                        
    print(f'{name} ({cid[:8]}):')
    print(f'  has_write_to_file={has_write}')
    if write_args:
        print(f'  TargetFile={write_args.get("TargetFile")}')
        print(f'  CodeLength={len(write_args.get("CodeContent", ""))}')
    if last_model_msg:
        print(f'  Last Model Msg: {last_model_msg[-200:]}')
    print('-'*50)
