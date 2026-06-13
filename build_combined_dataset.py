"""Build a combined TrOCR finetuning dataset from sample_1000 + MBumtiwiwiwi.

Output:
  output/combined_dataset/            flat folder of *.png images
  output/combined_transcribed.txt     tab-separated  <filename>\t<text>

Dedup is by image *content* (MD5). For images that exist in both sources the
sample_1000 (curated v2) copy + transcription wins.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "output" / "sample_1000"
SAMPLE_TXT = ROOT / "output" / "sample_1000_transcribed_v2.txt"
MB_DIR = ROOT / "output" / "MBumtiwiwiwi" / "umtiwiwiwi"
EXTRA_TXT = ROOT / "output" / "800_transcribed.txt"   # <docid>/line_XXX.png\ttext, rel to 3_lines
LINES_DIR = ROOT / "output" / "3_lines"

OUT_DIR = ROOT / "output" / "combined_dataset"
OUT_TXT = ROOT / "output" / "combined_transcribed.txt"
DROPPED_TXT = ROOT / "output" / "800_dropped.txt"     # audit log of rejected 800 lines

MARKER_ONLY = re.compile(r"^(\[[^\]]*\]\s*)+$")  # text that is nothing but [..] markers

POLISH_LETTERS = set("abcdefghijklmnopqrstuvwxyząćęłńóśźż")
RE_QQ = re.compile(r"\?{2,}")
RE_MIDWORD_Q = re.compile(r"[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]\?[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]")


def polish_reject_reason(text: str) -> str | None:
    """Return a reason string if the line should be REJECTED, else None.

    Rejects: empty / marker-only, only punctuation, <=2 letters, OCR garbage
    (replacement char, mid-word or repeated '?'), and non-Polish-alphabet letters.
    """
    t = text.strip()
    if not t or MARKER_ONLY.match(t):
        return "empty/marker-only"
    if "�" in t:
        return "replacement-char"
    if RE_QQ.search(t) or RE_MIDWORD_Q.search(t):
        return "garbled-?"
    letters = [c for c in t if c.isalpha()]
    if len(letters) <= 2:
        return "<=2 letters / punctuation-only"
    foreign = {c for c in letters if c.lower() not in POLISH_LETTERS}
    if foreign:
        return "non-polish-letter:" + "".join(sorted(foreign))
    return None


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_usable(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if MARKER_ONLY.match(t):
        return False
    return True


def parse_transcription(path: Path) -> dict[str, str]:
    """Return {line_filename: text} handling every schema variant seen in the data."""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fix the "missing comma between entries" corruption: a closing quote at
        # end of line directly followed by the next "key" line.
        fixed = re.sub(r'"\n(\s*")', '",\n\\1', raw)
        data = json.loads(fixed)

    result: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            result[k] = v
    elif isinstance(data, list):
        for i, item in enumerate(data, start=1):
            # {"line": "line_001.png", "transcription": "..."}
            if "line" in item and "transcription" in item:
                result[item["line"]] = item["transcription"]
            else:
                # ordered items with no filename -> map to line_{i:03d}.png
                text = item.get("text_best") or item.get("transcription") or item.get("text") or ""
                result[f"line_{i:03d}.png"] = text
    else:
        raise ValueError(f"Unexpected JSON root in {path}: {type(data)}")
    return result


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    seen_hashes: dict[str, str] = {}     # content hash -> final filename
    records: list[tuple[str, str]] = []  # (final filename, text)

    # ---- 1. sample_1000 (curated v2) --------------------------------------
    n_sample = 0
    with open(SAMPLE_TXT, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"{SAMPLE_TXT}:{ln} not tab-separated")
            name, text = line.split("\t", 1)
            src = SAMPLE_DIR / name
            if not src.exists():
                raise FileNotFoundError(f"sample_1000 missing image: {name}")
            if not is_usable(text):
                continue
            h = md5(src)
            if h in seen_hashes:
                continue  # internal dup (none expected)
            seen_hashes[h] = name
            shutil.copy2(src, OUT_DIR / name)
            records.append((name, text))
            n_sample += 1

    # ---- 2. MBumtiwiwiwi ---------------------------------------------------
    n_mb = 0
    n_dup = 0
    n_marker = 0
    n_nolabel = 0
    for sub in sorted(p for p in MB_DIR.iterdir() if p.is_dir()):
        tj = sub / "transcription.json"
        if not tj.exists():
            continue
        labels = parse_transcription(tj)
        for img in sorted(sub.glob("line_*.png")):
            text = labels.get(img.name)
            if text is None:
                n_nolabel += 1
                continue
            if not is_usable(text):
                n_marker += 1
                continue
            h = md5(img)
            if h in seen_hashes:
                n_dup += 1
                continue
            final = f"{sub.name}_{img.name}"  # e.g. 20251129155921036_p001_line_001.png
            seen_hashes[h] = final
            shutil.copy2(img, OUT_DIR / final)
            records.append((final, text))
            n_mb += 1

    # ---- 3. 800_transcribed.txt (raw OCR, needs validity filtering) -------
    n_extra = 0
    e_dup = 0
    e_missing = 0
    e_pathdup = 0
    e_reject: dict[str, int] = {}
    seen_extra_paths: set[str] = set()
    dropped_log: list[tuple[str, str, str]] = []  # (relpath, reason, text)
    with open(EXTRA_TXT, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"{EXTRA_TXT}:{ln} not tab-separated")
            rel, text = line.split("\t", 1)
            if rel in seen_extra_paths:
                e_pathdup += 1            # same image listed twice in the txt -> keep first
                continue
            seen_extra_paths.add(rel)
            src = LINES_DIR / rel
            if not src.exists():
                e_missing += 1
                continue
            reason = polish_reject_reason(text)
            if reason is not None:
                e_reject[reason.split(":")[0]] = e_reject.get(reason.split(":")[0], 0) + 1
                dropped_log.append((rel, reason, text))
                continue
            h = md5(src)
            if h in seen_hashes:
                e_dup += 1
                continue
            final = rel.replace("/", "_")  # 20251129..._p002/line_023.png -> ..._p002_line_023.png
            seen_hashes[h] = final
            shutil.copy2(src, OUT_DIR / final)
            records.append((final, text))
            n_extra += 1

    with open(DROPPED_TXT, "w", encoding="utf-8") as f:
        for rel, reason, text in dropped_log:
            f.write(f"{rel}\t{reason}\t{text}\n")

    # ---- 4. write combined txt --------------------------------------------
    records.sort(key=lambda r: r[0])
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for name, text in records:
            f.write(f"{name}\t{text}\n")

    print(f"sample_1000 kept     : {n_sample}")
    print(f"MBumtiwiwiwi kept    : {n_mb}")
    print(f"  dropped (dup img)  : {n_dup}")
    print(f"  dropped (marker)   : {n_marker}")
    print(f"  dropped (no label) : {n_nolabel}")
    print(f"800_transcribed kept : {n_extra}")
    print(f"  dropped (dup img)  : {e_dup}")
    print(f"  dropped (path dup) : {e_pathdup}")
    print(f"  dropped (no image) : {e_missing}")
    for r, c in sorted(e_reject.items()):
        print(f"  dropped ({r}) : {c}")
    print(f"  -> audit log       : {DROPPED_TXT}")
    print(f"TOTAL samples        : {len(records)}")
    print(f"Images folder        : {OUT_DIR}")
    print(f"Transcriptions file  : {OUT_TXT}")


if __name__ == "__main__":
    main()
