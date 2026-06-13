import sys, re, json
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch_file_list.txt', 'r', encoding='utf-8') as f:
    scratch_files = [line.strip().replace('\ufeff', '') for line in f if line.strip()]

with open('output/sample_1000_transcribed.txt', 'r', encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f if l.strip()]

suspicious_indices = []

for i, line in enumerate(lines):
    idx = i + 1
    parts = line.split('\t')
    fname = parts[0].strip()
    transcription = parts[1].strip() if len(parts) >= 2 else ''

    is_bad = False

    meta_patterns = [r'\[blank', r'\[MISSING', r'\[empty', r'\(blank', r'\(empty',
                     r'nearly empty', r'no text', r'\[no legible', r'illegible',
                     r'not legible', r'\[unclear', r'\[unreadable']
    for pat in meta_patterns:
        if re.search(pat, transcription, re.IGNORECASE):
            is_bad = True
            break

    error_words = ['error', 'failed', 'cannot', 'unable to', 'n/a']
    for ew in error_words:
        if ew.lower() in transcription.lower() and len(transcription) < 30:
            is_bad = True
            break

    words = transcription.split()
    if len(words) >= 4:
        for j in range(len(words)-2):
            if words[j].lower().strip('.,;:-') == words[j+1].lower().strip('.,;:-') == words[j+2].lower().strip('.,;:-') and words[j].lower().strip('.,;:-') not in ['']:
                is_bad = True
                break

    if re.match(r'^[\s\-]+$', transcription) and len(transcription) > 3:
        is_bad = True

    known_blanks = {402, 405, 406, 616}
    if transcription and len(transcription) < 3 and idx not in known_blanks:
        is_bad = True

    if is_bad:
        suspicious_indices.append(idx)

# Group into batches of 20
batches = []
batch_size = 20
for i in range(0, len(suspicious_indices), batch_size):
    batch = suspicious_indices[i:i+batch_size]
    batch_items = [(idx, scratch_files[idx-1]) for idx in batch]
    batches.append(batch_items)

print(f'Total bad: {len(suspicious_indices)}')
print(f'Number of batches: {len(batches)}')
print()

for b_idx, batch in enumerate(batches):
    print(f'=== Batch {b_idx+1} (lines {batch[0][0]}-{batch[-1][0]}) ===')
    for idx, fname in batch:
        print(f'{idx}: {fname}')
    print()

# Save as JSON for easy reading
with open('bad_lines.json', 'w', encoding='utf-8') as f:
    json.dump([(idx, scratch_files[idx-1]) for idx in suspicious_indices], f, ensure_ascii=False, indent=2)
print('Saved to bad_lines.json')
