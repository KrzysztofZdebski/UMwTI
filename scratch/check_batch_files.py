import os

files = sorted([f for f in os.listdir('output') if 'retrans_batch' in f])
print(f"Retrans batch files in output: {files}")
for f in files:
    p = os.path.join('output', f)
    print(f"  {f}: size={os.path.getsize(p)} bytes")
