from pathlib import Path
from collections import Counter

split_dir = Path("data/split/train")
counts = Counter({d.name: len(list(d.iterdir())) for d in split_dir.iterdir() if d.is_dir()})
for cls, count in counts.most_common():
    print(f"{cls}: {count}")