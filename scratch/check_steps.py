import os
import json

cids = {
    'Batch 1': 'e69f92b1-c600-4db8-b79b-96a30bdd1921',
    'Batch 4': 'a6978a81-5e3e-4bf3-a940-7ce996db0291',
    'Batch 6': '5f84d078-0aa1-4012-9b19-c56b2822e01f',
    'Batch 7': '9fc5f8e1-3157-4ce0-9204-b296606edf45',
    'Batch 8': '7bfc1eea-4122-4c7e-a8c3-c5856133399d',
    'Batch 9': '0938c05d-fb20-4e7f-a56d-c55e1b2288c1'
}

for name, cid in sorted(cids.items()):
    log_path = f'C:/Users/KMZde/.gemini/antigravity/brain/{cid}/.system_generated/logs/transcript.jsonl'
    if not os.path.exists(log_path):
        print(f'{name}: log missing')
        continue
    
    with open(log_path, 'r', encoding='utf-8') as f:
        steps = [json.loads(line) for line in f]
        
    print(f"\n=== {name} ({cid[:8]}) total steps={len(steps)} ===")
    for i in range(max(0, len(steps)-3), len(steps)):
        step = steps[i]
        print(f"  Step {step.get('step_index')}: source={step.get('source')}, type={step.get('type')}, status={step.get('status')}")
        if step.get('tool_calls'):
            print(f"    Tool Calls: {[{'name': tc.get('name'), 'args': {k: v for k, v in tc.get('args', {}).items() if k != 'CodeContent'}} for tc in step['tool_calls']]}")
        if step.get('content') and step.get('type') == 'PLANNER_RESPONSE':
            cleaned = step.get('content')[:150].encode('ascii', 'ignore').decode('ascii')
            print(f"    Content: {cleaned}")
