# Independent prose reviews

The prose was reviewed three times by fresh reviewers (separate model instances that
had not seen the paper or each other's reviews). Each read only the PDF text and
scored how human-written it reads (10 = indistinguishable from a careful researcher),
quoting passages that sounded machine-generated. Every flagged passage was revised;
facts and numbers were never changed.

| Round | Score | Main flags | What changed |
|---|---|---|---|
| 1 | 7 / 10 | slogan-style conclusion and checks opener, "X, not Y" lines, stacked lists, "a single" used three times | conclusion and abstract rewritten, run-in headings made descriptive, slogans replaced |
| 2 | 7.5 / 10 | "instrument" metaphor repeated, unsupported "cheap", templated "X is caught by Y" sentences, conclusion overclaimed ("caught every problem") | checks section restructured, overclaim corrected to exclude the two problems that need live runs |
| 3 | 7 / 10 | signposted "lesson" sentence, "could never" overstatement, recap sentence with neat parallel clauses | all six flags revised |

Reviewers consistently kept: the opening fraud-team scenario, the admission that our own
oracle had a wrong goal, the "double charge and all" aside, and the concrete task-64 detail.
Automated checks (`python paper/style_check.py`) pass on the final text.
