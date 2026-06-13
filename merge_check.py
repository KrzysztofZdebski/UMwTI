import sys, os
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch_file_list.txt', 'r', encoding='utf-8') as f:
    scratch_files = [line.strip().replace('\ufeff', '') for line in f if line.strip()]

# Load master file as dict
master = {}
with open('output/sample_1000_transcribed.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            master[parts[0].strip()] = parts[1].strip()
        elif len(parts) == 1 and parts[0].strip():
            master[parts[0].strip()] = ''

print('Master file entries:', len(master))

# Load part files as dict
part_data = {}
output_dir = 'output'
part_files = sorted([f for f in os.listdir(output_dir) if f.startswith('part_')])
for pf in part_files:
    path = os.path.join(output_dir, pf)
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts2 = line.strip().split('\t')
            if len(parts2) >= 2 and parts2[0].strip():
                part_data[parts2[0].strip()] = parts2[1].strip()
            elif len(parts2) == 1 and parts2[0].strip():
                part_data[parts2[0].strip()] = ''

print('Part files unique entries:', len(part_data))

# Find which scratch files are only in parts (not in master)
only_in_parts = [(i+1, f) for i, f in enumerate(scratch_files) if f not in master and f in part_data]
print('Only in parts (missing from master):', len(only_in_parts))
for idx, f in only_in_parts:
    transcription = part_data.get(f, '')
    print('  Line ' + str(idx) + ': ' + f + ' -> ' + repr(transcription))

missing_from_both = [i+1 for i, f in enumerate(scratch_files) if f not in master and f not in part_data]
print('Missing from both:', missing_from_both)

# Build the complete merged output in scratch_file_list order
print('\nBuilding merged output...')
merged = {}
# Priority: master > part_data
for f in scratch_files:
    if f in master:
        merged[f] = master[f]
    elif f in part_data:
        merged[f] = part_data[f]
    else:
        merged[f] = '[MISSING]'

missing_files = [f for f in scratch_files if merged[f] == '[MISSING]']
print('Files that would be MISSING in merged output:', len(missing_files))

# Write merged output
out_path = 'output/sample_1000_MERGED.txt'
with open(out_path, 'w', encoding='utf-8') as fout:
    for f in scratch_files:
        transcription = merged.get(f, '[MISSING]')
        fout.write(f + '\t' + transcription + '\n')

print('Merged output written to:', out_path)
print('Total lines written:', len(scratch_files))
