import sys, os, re
sys.stdout.reconfigure(encoding='utf-8')

# Load master file as ordered list
with open('scratch_file_list.txt', 'r', encoding='utf-8') as f:
    scratch_files = [line.strip().replace('\ufeff', '') for line in f if line.strip()]

with open('output/sample_1000_transcribed.txt', 'r', encoding='utf-8') as f:
    master_lines = [l.rstrip('\n') for l in f if l.strip()]

master = {}
for line in master_lines:
    parts = line.split('\t')
    fname = parts[0].strip()
    transcription = parts[1].strip() if len(parts) >= 2 else ''
    master[fname] = transcription

print(f'Master file loaded: {len(master)} entries')

# Load all retrans batch files
retrans = {}
batch_files = sorted([f for f in os.listdir('output') if f.startswith('retrans_batch') or f == 'retrans_fix.txt'])
print(f'Found retrans batch files: {batch_files}')

for bf in batch_files:
    path = os.path.join('output', bf)
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if '\t' in line:
                fname, transcription = line.split('\t', 1)
                fname = fname.strip()
                transcription = transcription.strip()
                if fname in [sf for sf in scratch_files]:
                    retrans[fname] = transcription
                    count += 1
    print(f'  {bf}: loaded {count} corrections')

print(f'Total corrections loaded: {len(retrans)}')

# Apply corrections
applied = 0
for fname, new_trans in retrans.items():
    if fname in master:
        old = master[fname]
        if old != new_trans:
            master[fname] = new_trans
            applied += 1

print(f'Corrections applied: {applied}')

# Write updated master file in scratch_file_list order
out_path = 'output/sample_1000_transcribed_v2.txt'
with open(out_path, 'w', encoding='utf-8') as fout:
    for fname in scratch_files:
        transcription = master.get(fname, '')
        fout.write(fname + '\t' + transcription + '\n')

print(f'Updated master written to: {out_path}')

# Run audit on the new file
def audit_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    
    known_blanks = {
        402, 405, 406, 616,
        # Legitimately short content (single marks, letters, punctuation)
        10, 25, 28, 39, 40, 46, 52, 58, 76, 77, 97, 98,
        200, 223, 347, 424, 673, 684, 686, 728, 784, 831, 835, 848,
        # Single dash lines (legitimate punctuation/separator)
        158, 163, 195, 672,
    }
    bad = []
    
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
        
        words = transcription.split()
        if len(words) >= 4:
            for j in range(len(words)-2):
                if words[j].lower().strip('.,;:-') == words[j+1].lower().strip('.,;:-') == words[j+2].lower().strip('.,;:-') and words[j].lower().strip('.,;:-'):
                    is_bad = True
                    break
        
        if re.match(r'^[\s\-]+$', transcription) and len(transcription) > 3:
            is_bad = True
        
        if transcription and len(transcription) < 3 and idx not in known_blanks:
            is_bad = True
        
        if is_bad:
            bad.append((idx, fname, transcription))
    
    return bad

bad_after = audit_file('output/sample_1000_transcribed_v2.txt')
print(f'\nRemaining bad lines after corrections: {len(bad_after)}')
for idx, fname, trans in bad_after[:20]:
    print(f'  Line {idx}: {fname}')
    print(f'    -> {repr(trans[:80])}')
