"""Objective prose checks for main.tex (body text, excluding floats and math).

    python paper/style_check.py
"""
import re
import statistics
from collections import Counter

tex = open("paper/main.tex").read()
body = tex[tex.index("\\begin{abstract}"):tex.index("\\bibliography{")]
t = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{(table|figure)\}", " ", body, flags=re.S)
t = re.sub(r"\\(cite|citet|citep)\{[^}]*\}", "X", t)
t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", t)
t = re.sub(r"[{}~$]", " ", t)
flat = " ".join(t.split())
flat = re.sub(r"([.!?])('')", r"\1", flat)            # sentence ending inside a closing quote
flat = re.sub(r"([.!?])\)", r"\1", flat)              # sentence ending inside a parenthesis
sents = [s for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.split()) > 2]
lengths = [len(s.split()) for s in sents]
tells = ["delve", "notably", "crucial", "moreover", "furthermore", "additionally", "in summary",
         "in conclusion", "leverag", "robust", "comprehensive", "landscape", "pivotal", "underscore",
         "showcase", "seamless", "intricate", "realm", "tapestry", "paramount", "it is important to note",
         "plays a key role", "groundbreaking", "shed light", "multifaceted", "holistic", "meticulous"]
found = {w: t.lower().count(w) for w in tells if w in t.lower()}
top_opener, top_count = Counter(s.split()[0] for s in sents).most_common(1)[0]
checks = [
    ("no stock machine-text vocabulary", not found, found or "none"),
    ("no em-dashes", body.count("---") == 0, body.count("---")),
    ("varied sentence length (stdev >= 8 words)", statistics.stdev(lengths) >= 8, round(statistics.stdev(lengths), 1)),
    ("no sentence over 45 words", max(lengths) <= 45, max(lengths)),
    ("no sentence opener used for >15% of sentences", top_count / len(sents) <= 0.15, f"{top_opener!r} {top_count}/{len(sents)}"),
    ("mean sentence length 15-25 words (readable academic prose)", 15 <= statistics.mean(lengths) <= 25, round(statistics.mean(lengths), 1)),
]
for name, ok, val in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {val}")
print("\nALL STYLE CHECKS PASS" if all(ok for _, ok, _ in checks) else "\nSOME STYLE CHECKS FAIL")
