import os

output_dir = 'output'
part_files = sorted([f for f in os.listdir(output_dir) if f.startswith('part_') and f.endswith('.txt')])
print(f"Found {len(part_files)} part files:")
for pf in part_files:
    p = os.path.join(output_dir, pf)
    print(f"  {pf}: size={os.path.getsize(p)} bytes")
