"""Check every refs.bib entry against a primary record and print PASS/FAIL per field.

    python paper/verify_sources.py
"""
import json, re, subprocess, urllib.request

def bib():
    out = {}
    for m in re.finditer(r"@\w+\{(\w+),(.*?)\n\}", open("paper/refs.bib").read(), re.S):
        out[m.group(1)] = {k: v for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\n", m.group(2) + "\n")}
    return out

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "refcheck/1.0 (mailto:none)"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")

def norm(x):
    return re.sub(r"[^a-z0-9]", "", x.lower())

B = bib()
ok_all = True
def check(key, label, cond):
    global ok_all
    ok_all &= cond
    print(f"{'PASS' if cond else 'FAIL'}  {key:26s} {label}")

ARXIV = {"yao2025taubench": "2406.12045", "barres2025tau2": "2506.07982", "zhu2025abc": "2507.02825",
         "trivedi2024appworld": "2407.18901", "jimenez2024swebench": "2310.06770", "zhou2024webarena": "2307.13854",
         "chen2026unreliable": "2610.16067", "he2026sscbench": "2610.11514", "shao2026protocol": "2607.22368"}
for key, aid in ARXIV.items():
    xml = get(f"https://export.arxiv.org/api/query?id_list={aid}")
    entry = xml.split("<entry>")[1]
    title = re.search(r"<title>(.*?)</title>", entry, re.S).group(1)
    authors = [norm(a.split()[-1]) for a in re.findall(r"<name>(.*?)</name>", entry)]
    bib_title = re.sub(r"\$.*?\$", "", B[key]["title"])
    check(key, f"title (arXiv:{aid})", norm(bib_title).replace("bench", "") in norm(title).replace("bench", ""))
    bib_auth = [norm(a.split(",")[0]) for a in B[key]["author"].split(" and ")]
    check(key, f"authors in order ({len(authors)})", bib_auth == authors)

DOI = {"messick1995validity": "10.1037/0003-066X.50.9.741", "demillo1978hints": "10.1109/C-M.1978.218136",
       "jia2011mutation": "10.1109/TSE.2010.62"}
for key, doi in DOI.items():
    m = json.loads(get(f"https://api.crossref.org/works/{doi}"))["message"]
    b = B[key]
    check(key, f"title (Crossref {doi})", norm(b["title"]) in norm(m["title"][0]) or norm(m["title"][0]) in norm(b["title"]))
    check(key, "journal", norm(b["journal"]) == norm(m["container-title"][0]))
    check(key, "volume/number/pages", (b["volume"], b["number"], b["pages"].replace("--", "-")) == (m["volume"], m["issue"], m["page"]))
    check(key, "year", int(b["year"]) == m["issued"]["date-parts"][0][0])
    check(key, "authors", [norm(a.split(",")[0]) for a in b["author"].split(" and ")] == [norm(a["family"]) for a in m["author"]])

# McKeeman (Digital Technical Journal has no DOI): read the article PDF itself.
pdf = "/tmp/mck.pdf"
subprocess.run(["curl", "-sS", "-L", "-o", pdf,
                "https://www.cs.tufts.edu/comp/150FP/archive/bill-mckeeman/DifferentailTesting.pdf"], check=True)
text = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
b = B["mckeeman1998differential"]
check("mckeeman1998differential", "title and author on first page", "Differential Testing" in text[:200] and "McKeeman" in text[:200])
check("mckeeman1998differential", "journal header 'Digital Technical Journal' + 'Vol. 10 No. 1'", "Digital Technical Journal" in text and "Vol. 10 No. 1" in text)
check("mckeeman1998differential", "first page number 100, 8 pages -> 100--107", re.search(r"^100$", text, re.M) is not None and pages == 8 and b["pages"] == "100--107")
print("\nALL SOURCES VERIFIED" if ok_all else "\nSOME CHECKS FAILED")
