"""Check a Markdown file against the mechanical rules of ASD-STE100.

ASD-STE100 is Simplified Technical English. It limits the words and the verb
forms that a technical document can use. This script checks the rules that a
program can check. A person must still check the other rules, for example the
approved vocabulary and the maximum of three words in a noun cluster.

The rules this script checks:

    R1  A description has 25 words or less in a sentence.
    R2  An instruction has 20 words or less in a sentence.
    R3  Use only these verb forms: infinitive, imperative, simple present,
        simple past, simple future, and the past participle as an adjective.
        Do not use the present perfect, the past perfect, or a continuous form.
    R4  Use "must", "can", or "will". Do not use "should", "may", "might",
        "could", or "would".
    R5  Do not use a contraction or a possessive with an apostrophe.
    R6  Do not use "there is", "there are", "there was", or "there were".
    R7  An "-ing" word must be a technical noun, for example "embedding".
    R9  A paragraph has six sentences or less.

Two checks give a note and not an error. They are advice and not rules:

    N1  Write in the active voice where the actor is known.
    N2  "Which" often makes a long clause. A second sentence is usually better.

A paragraph is a group of adjacent prose lines. A blank line, a heading, and
the first line of a list item each start a new paragraph. The script joins the
lines of a paragraph before it counts the sentences, because the file wraps a
paragraph at 78 columns and a wrapped sentence is not two sentences.

Run:

    python3 check_ste100.py document.md
    python3 check_ste100.py document.md --ing-nouns domain_terms.txt
    python3 check_ste100.py document.md --list

The script uses only the Python standard library. Use --ing-nouns one or more
times to allow technical nouns from a plain-text word list. Put one word on
each line. Lines that start with # are comments.
"""
from __future__ import annotations

import argparse
import re
import sys

GENERIC_ING_NOUNS = {
    "according", "anything", "building", "checking", "closing", "copying",
    "drawing", "during", "everything", "existing", "fitting", "following",
    "learning", "listing", "meaning", "meeting", "nothing", "opening",
    "printing", "reading", "reasoning", "remaining", "running", "setting",
    "settings", "shipping", "something", "sorting", "string", "strings",
    "testing", "thinking", "warning", "working", "writing",
}

MODALS = ("should", "may", "might", "could", "would")
CONTRACTION = re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I)
THERE = re.compile(r"\bthere\s+(is|are|was|were)\b", re.I)
ING = re.compile(r"\b([a-z]+ing)\b")
WHICH = re.compile(r"\bwhich\b", re.I)
PERFECT = re.compile(
    r"\b(has|have|had)\s+(\w+ed|been|being|got|made|given|taken|shown|written|"
    r"built|found|kept|held|sent|put|read|set|done|become|begun|chosen)\b",
    re.I)
CONTINUOUS = re.compile(
    r"\b(is|are|was|were|be|been|being|am)\s+(\w+ing)\b", re.I)
PASSIVE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+(\w+ed|built|made|given|shown|taken|"
    r"kept|held|read|set|put|done|found|left|sent|written|drawn|known)\b", re.I)
LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s+")
# A continuation line of a list item is prose. A command or an assignment is
# not, even when a list item comes before it.
CODE_HINT = re.compile(
    r"^(\$|[A-Za-z_][A-Za-z0-9_]*=|\./|-{1,2}[a-z]|python|cc\b|gcc\b|"
    r"clang\b|pip\b|curl\b|nproc\b|export\b|cd\b|ls\b|cat\b|make\b)")
SENTENCE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9`\"'(])")
CODE = re.compile(r"`[^`]*`")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'./-]*")
IMPERATIVE = {"run", "use", "set", "check", "show", "add", "copy", "write",
              "build", "load", "keep", "make", "open", "close", "do", "read",
              "give", "put", "start", "stop", "see", "compare", "type",
              "export", "install", "delete", "remove", "change", "enter"}

LABEL_RULE = {
    "not approved verb form (perfect)": "R3",
    "not approved verb form (continuous)": "R3",
    "contraction or possessive": "R5",
    "there is / there are": "R6",
    "-ing word that is not a known noun": "R7",
    "passive voice": "N1",
    "which": "N2",
}


def prose_lines(lines):
    """Return (number, text) for each line that is not code and not a table."""
    out = []
    fence = False
    in_list = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fence = not fence
            in_list = False
            continue
        if fence:
            continue
        if line.lstrip().startswith("|"):
            continue
        if not line.strip():
            in_list = False
            out.append((i, line))
            continue
        if LIST_ITEM.match(line):
            in_list = True
            out.append((i, line))
            continue
        if line.startswith("    ") or line.startswith("\t"):
            if in_list and not CODE_HINT.match(line.strip()):
                out.append((i, line))
            continue
        if line.lstrip().startswith("$ "):
            in_list = False
            continue
        out.append((i, line))
    return out


def paragraphs(prose):
    """Return (first line number, text) for each paragraph of the prose."""
    out = []
    buf = []
    start = 0

    def flush():
        if buf:
            out.append((start, " ".join(buf)))

    for num, line in prose:
        s = line.strip()
        if not s:
            flush()
            buf, start = [], 0
            continue
        if s.startswith("#"):
            flush()
            buf, start = [], 0
            out.append((num, s.lstrip("# ").strip()))
            continue
        if LIST_ITEM.match(s):
            flush()
            buf, start = [LIST_ITEM.sub("", s)], num
            continue
        if not buf:
            start = num
        buf.append(s)
    flush()
    return out


def sentence_words(text):
    """Return the word count of each sentence of one paragraph."""
    out = []
    # A bold lead-in such as "**Copy the file.** The ..." ends a sentence. The
    # emphasis marker hides the full stop from the split, so remove the
    # markers first. A marker is not a word, so the count does not change.
    for part in SENTENCE.split(text.replace("**", " ").replace("__", " ")):
        part = part.strip()
        if part:
            out.append((len(WORD.findall(CODE.sub(" CODE ", part))), part))
    return out


def is_instruction(text):
    """Return True when a sentence looks like an instruction."""
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return False
    if text.rstrip().endswith(":"):
        return True
    return words[0].lower() in IMPERATIVE


def check(path, show_list=False, ing_nouns=()):
    known_ing_nouns = GENERIC_ING_NOUNS | {word.lower() for word in ing_nouns}
    with open(path, encoding="utf-8") as source:
        lines = source.read().split("\n")
    prose = prose_lines(lines)
    bad = []

    # The word checks apply to one line at a time.
    for num, text in prose:
        plain = CODE.sub(" CODE ", text)
        for label, pattern in (
                ("not approved verb form (perfect)", PERFECT),
                ("not approved verb form (continuous)", CONTINUOUS),
                ("contraction or possessive", CONTRACTION),
                ("there is / there are", THERE),
                ("-ing word that is not a known noun", ING),
                ("passive voice", PASSIVE),
                ("which", WHICH)):
            if label == "not approved verb form (continuous)":
                hits = [m.group(2) for m in CONTINUOUS.finditer(plain)
                        if m.group(2).lower() not in known_ing_nouns]
            elif label == "-ing word that is not a known noun":
                hits = [h for h in ING.findall(plain)
                        if h.lower() not in known_ing_nouns]
            else:
                hits = [m.group(0) for m in pattern.finditer(plain)]
            for hit in hits:
                bad.append((num, LABEL_RULE[label], label, hit, text.strip()[:96]))
        for hit in re.findall(r"\b(%s)\b" % "|".join(MODALS), plain, re.I):
            bad.append((num, "R4", "not approved modal", hit, text.strip()[:96]))

    # The sentence checks apply to a whole paragraph, because the file wraps a
    # paragraph and a wrapped sentence is not two sentences.
    for start, text in paragraphs(prose):
        sw = sentence_words(text)
        if len(sw) > 6:
            bad.append((start, "R9", "paragraph has %d sentences (limit 6)"
                        % len(sw), "%d" % len(sw), text[:96]))
        for n, sent in sw:
            limit = 20 if is_instruction(sent) else 25
            if n > limit:
                bad.append((start, "R2" if limit == 20 else "R1",
                            "sentence has %d words (limit %d)" % (n, limit),
                            "%d words" % n, sent[:96]))

    bad.sort()
    errors = [b for b in bad if b[1][0] == "R"]
    notes = [b for b in bad if b[1][0] == "N"]
    for num, rule, label, hit, text in (bad if show_list else errors):
        print("%5d  %-3s  %-42s %-12s %s" % (num, rule, label[:42], hit[:12], text))
    print("\n%s: %d lines, %d errors, %d notes"
          % (path, len(lines), len(errors), len(notes)))
    return 1 if errors else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--list", action="store_true", help="show the notes too")
    ap.add_argument("--ing-nouns", action="append", default=[], metavar="FILE",
                    help="plain-text list of accepted technical nouns; can repeat")
    args = ap.parse_args()
    ing_nouns = set()
    for path in args.ing_nouns:
        with open(path, encoding="utf-8") as source:
            for line in source:
                word = line.partition("#")[0].strip().lower()
                if word:
                    ing_nouns.add(word)
    return check(args.path, args.list, ing_nouns)


if __name__ == "__main__":
    sys.exit(main())
