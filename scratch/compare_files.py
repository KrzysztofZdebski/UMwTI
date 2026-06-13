import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('output/sample_1000_transcribed.txt', 'r', encoding='utf-8') as f:
    t1 = [l.strip() for l in f if l.strip()]
    
with open('output/sample_1000_MERGED.txt', 'r', encoding='utf-8') as f:
    t2 = [l.strip() for l in f if l.strip()]
    
print(f"transcribed lines: {len(t1)}")
print(f"merged lines: {len(t2)}")

diff = 0
for idx, (l1, l2) in enumerate(zip(t1, t2)):
    if l1 != l2:
        diff += 1
        if diff <= 10:
            print(f"Line {idx+1} diff:")
            print(f"  transcribed: {l1}")
            print(f"  merged:      {l2}")
            
print(f"Total differences: {diff}")
