import json

with open('bad_lines.json', 'r', encoding='utf-8') as f:
    bad_lines = json.load(f)

# Group into batches of 20
batch_size = 20
batches = []
for i in range(0, len(bad_lines), batch_size):
    batches.append(bad_lines[i:i+batch_size])

output_prompts = []
for b_idx, batch in enumerate(batches):
    if b_idx == 9: # Batch 10
        continue
    
    prompt = f"You must RE-TRANSCRIBE exactly {len(batch)} handwritten line images. These were previously transcribed incorrectly (hallucinated repeated words or too short). You MUST view each image carefully and write what you actually see.\n\nIMAGE DIRECTORY: c:\\Users\\KMZde\\Desktop\\AGH\\UMwTI\\output\\sample_1000\\\n\nIMPORTANT RULES:\n- Do NOT repeat words. If you find yourself writing the same word 3+ times, STOP and look at the image again.\n- Transcribe EXACTLY what you see in the handwritten image.\n- Keep Polish diacritics: ą, ć, ę, ł, ń, ó, ś, ź, ż\n- If a line truly has only 1-3 characters, that is okay, but double-check.\n- If an image is genuinely blank/empty, write just the filename with a tab and nothing after.\n\nFILES TO RE-TRANSCRIBE (use view_file for each image):\n"
    for idx, fname in batch:
        prompt += f"{idx}: {fname}\n"
        
    prompt += f"\nOUTPUT: Save results to c:\\Users\\KMZde\\Desktop\\AGH\\UMwTI\\output\\retrans_batch{b_idx+1}.txt\nFormat: one line per image = <filename>TAB<transcription>\n\nWhen done, send me a message with the complete list of your {len(batch)} transcriptions."
    
    output_prompts.append({
        'batch_num': b_idx + 1,
        'role': f"Re-transcriber Batch {b_idx+1}",
        'prompt': prompt
    })

with open('scratch/prompts.json', 'w', encoding='utf-8') as f:
    json.dump(output_prompts, f, ensure_ascii=False, indent=2)
print("Saved prompts to scratch/prompts.json successfully.")
