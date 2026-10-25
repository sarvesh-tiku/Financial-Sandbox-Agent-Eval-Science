# Reference verification

Each bibliography entry in `refs.bib` was checked against a primary record.

| Key | Checked against | Result |
|---|---|---|
| yao2025taubench | arXiv:2406.12045 abstract page | title and 4 authors match; no venue listed, cited as arXiv preprint |
| barres2025tau2 | arXiv:2506.07982 abstract page | title and 5 authors match |
| zhu2025abc | arXiv:2507.02825 abstract page | title and 25 authors match; abstract states "TAU-bench counts empty responses as successful" |
| trivedi2024appworld | arXiv:2407.18901 abstract page | title and 9 authors match; comments field "ACL'24 Camera Ready" |
| jimenez2024swebench | arXiv:2310.06770 abstract page | title and 7 authors match; venue ICLR 2024 |
| zhou2024webarena | arXiv:2307.13854 (authors), DBLP (ICLR 2024) | title and 12 authors match |
| messick1995validity | Crossref, DOI 10.1037/0003-066X.50.9.741 | American Psychologist 50(9):741-749, Sept 1995 |
| demillo1978hints | Crossref, DOI 10.1109/C-M.1978.218136 | Computer 11(4):34-41, April 1978 |
| jia2011mutation | Crossref, DOI 10.1109/TSE.2010.62 | IEEE TSE 37(5):649-678, 2011 |
| mckeeman1998differential | Original article PDF (journal pages) | Digital Technical Journal 10(1), pp. 100-107 (8 pages, header on every page) |

Reproduce with `python paper/verify_sources.py` (arXiv API, Crossref API, and the
McKeeman article PDF; prints PASS/FAIL per field). Prose checks: `python paper/style_check.py`.
