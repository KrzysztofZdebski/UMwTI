import sys, re
sys.stdout.reconfigure(encoding='utf-8')

with open('output/sample_1000_transcribed.txt', 'r', encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f if l.strip()]

suspicious_indices = []

for i, line in enumerate(lines):
    idx = i + 1
    parts = line.split('\t')
    fname = parts[0].strip()
    transcription = parts[1].strip() if len(parts) >= 2 else ''

    is_bad = False

    # META comments
    meta_patterns = [r'\[blank', r'\[MISSING', r'\[empty', r'\(blank', r'\(empty',
                     r'nearly empty', r'no text', r'\[no legible', r'illegible',
                     r'not legible', r'\[unclear', r'\[unreadable']
    for pat in meta_patterns:
        if re.search(pat, transcription, re.IGNORECASE):
            is_bad = True
            break

    # Error words
    error_words = ['error', 'failed', 'cannot', 'unable to', 'n/a']
    for ew in error_words:
        if ew.lower() in transcription.lower() and len(transcription) < 30:
            is_bad = True
            break

    # Word repetitions (3+ consecutive same words)
    words = transcription.split()
    if len(words) >= 4:
        for j in range(len(words)-2):
            if words[j].lower().strip('.,;:-') == words[j+1].lower().strip('.,;:-') == words[j+2].lower().strip('.,;:-') and words[j].lower().strip('.,;:-') not in ['']:
                is_bad = True
                break

    # Repeated dashes only
    if re.match(r'^[\s\-]+$', transcription) and len(transcription) > 3:
        is_bad = True

    # Very short (< 3 chars) but not known blank images (402, 405, 406, 616)
    known_blanks = {402, 405, 406, 616}
    if transcription and len(transcription) < 3 and idx not in known_blanks:
        is_bad = True

    if is_bad:
        suspicious_indices.append(idx)

print('Bad lines count:', len(suspicious_indices))
print('Bad line indices:', suspicious_indices)
