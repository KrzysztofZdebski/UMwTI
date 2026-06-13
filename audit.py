import sys, re
sys.stdout.reconfigure(encoding='utf-8')

with open('output/sample_1000_transcribed.txt', 'r', encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f if l.strip()]

print(f'Total lines: {len(lines)}')
print()

suspicious = []

for i, line in enumerate(lines):
    idx = i + 1
    parts = line.split('\t')
    fname = parts[0].strip()
    transcription = parts[1].strip() if len(parts) >= 2 else ''

    flags = []

    # 1. Empty transcription (already know 4 are blank images - mark separately)
    if not transcription:
        flags.append('EMPTY')

    # 2. Contains English meta-comments that shouldn't be in Polish text
    meta_patterns = [
        r'\[blank', r'\[MISSING', r'\[empty', r'\(blank', r'\(empty',
        r'nearly empty', r'no text', r'\[no legible', r'illegible',
        r'not legible', r'\[unclear', r'\[unreadable',
    ]
    for pat in meta_patterns:
        if re.search(pat, transcription, re.IGNORECASE):
            flags.append('META_COMMENT: ' + transcription[:60])
            break

    # 3. Very short transcriptions (less than 3 chars, not blank images)
    if transcription and len(transcription) < 3:
        flags.append('VERY_SHORT: ' + repr(transcription))

    # 4. Contains only punctuation or single characters
    if transcription and re.match(r'^[\W\s]{1,4}$', transcription):
        flags.append('PUNCT_ONLY: ' + repr(transcription))

    # 5. Contains obvious error text / English phrases in otherwise Polish context
    error_words = ['error', 'failed', 'cannot', 'unable to', 'transcription not', 'n/a', 'N/A']
    for ew in error_words:
        if ew.lower() in transcription.lower():
            flags.append('ERROR_WORD: ' + transcription[:60])
            break

    # 6. Ends mid-word or mid-sentence with a hyphen at a suspicious position
    # (could indicate cut-off, but hyphens are common in Polish cursive line breaks)
    # Only flag if very short + hyphen
    if transcription.endswith('-') and len(transcription) < 10:
        flags.append('SHORT_HYPHEN: ' + repr(transcription))

    # 7. Contains non-Polish characters that suggest OCR confusion or copy errors
    # Flag lines with excessive non-latin characters (excluding Polish diacritics)
    non_latin = re.sub(r'[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9\s\.,;:\-\'"!?()„"…–\[\]\/]', '', transcription)
    if len(non_latin) > 3:
        flags.append('NON_LATIN_CHARS: ' + repr(non_latin[:20]) + ' in: ' + transcription[:60])

    # 8. Repetitions (e.g. "które które które" - likely hallucination)
    words = transcription.split()
    if len(words) >= 4:
        # Check for 3+ consecutive word repetitions
        for j in range(len(words)-2):
            if words[j] == words[j+1] == words[j+2]:
                flags.append('WORD_REPETITION: ' + transcription[:60])
                break

    # 9. Suspiciously long lines (might include extra text)
    if len(transcription) > 200:
        flags.append('VERY_LONG: ' + str(len(transcription)) + ' chars')

    if flags:
        suspicious.append((idx, fname, transcription, flags))

print(f'Suspicious lines found: {len(suspicious)}')
print()
for idx, fname, trans, flags in suspicious:
    print(f'Line {idx}: {fname}')
    print(f'  Transcription: {repr(trans[:120])}')
    print(f'  Flags: {flags}')
    print()
