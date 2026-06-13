import json

with open(r"C:\Users\KMZde\.gemini\antigravity\brain\b37f2338-058b-4c97-8a8d-e54eaf6d7175\.system_generated\logs\transcript.jsonl", "r", encoding="utf-8") as f:
    lines = list(f)

out_lines = []
for idx, line in enumerate(lines):
    try:
        obj = json.loads(line)
        thinking = obj.get("thinking", "")
        if thinking:
            out_lines.append(f"=== STEP {obj.get('step_index')} ===")
            out_lines.append(thinking)
    except Exception as e:
         pass

with open(r"C:\Users\KMZde\Desktop\AGH\UMwTI\scratch\find_prev_all_thinking.txt", "w", encoding="utf-8") as f_out:
    f_out.write("\n".join(out_lines))
