"""The hard tier of the quality suite (scripts/quality_suite.py --tier hard).

The basic tier (suite.py) sits near its ceiling on a good model; it finds a
broken form (template, tool parse, think part, long context) at once, but a
change as small as the NVFP4 experts against RQ6/RQ8 (about 1% of
perplexity) moves few of its items. These items are harder, so such a
change flips some of them:

    hmath      multi-step problems; every key computed here (brute force
               where it can be)
    hreason    logic, dates, state search; the keys computed or checked here
    hcode      algorithms with edge-case unit tests
    htools     agent tasks: the model calls tools over several turns against
               a scripted environment (files, commands, a calendar, mail,
               rates); graded on the final answer and on what it did
    hinstruct  several checkable constraints at once
    hneedle    three facts at three depths of a long text, to find and
               combine (--needle-hard lengths, 32K and 128K by default)

An htools item has env(): a fresh environment with tools(name, args) ->
str and a state the grader reads; grade(msg, env, trace).
"""
from __future__ import annotations

import ast
import datetime
import itertools
import json
import math
import re
from collections import deque
from fractions import Fraction

from suite import (S, boxed_or_last, extract_code, fn, num_grader, run_tests, user, words)

MATH_TAIL = " Think it through carefully. Give the final answer as a single number on the last line."


# ---------------------------------------------------------------- hmath

def _primes(n):
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\0\0"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n + 1) if s[i]]


def _ndiv(n):
    return sum(1 for d in range(1, n + 1) if n % d == 0)


def _fact_zeros(n):
    z, p = 0, 5
    while p <= n:
        z += n // p
        p *= 5
    return z


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _hmath():
    items = []

    def add(q, ans):
        items.append({"id": "hmath_%02d" % (len(items) + 1), "cat": "hmath",
                      "messages": user(q + MATH_TAIL), "grade": num_grader(ans)})
    add("How many integers from 1 to 1000 inclusive are divisible by 3 or by 5, but not by 15?",
        sum(1 for n in range(1, 1001) if (n % 3 == 0 or n % 5 == 0) and n % 15))
    add("What is the sum of the decimal digits of 2^50?", sum(map(int, str(2 ** 50))))
    add("In how many distinct ways can the letters of the word BANANA be arranged?",
        len(set(itertools.permutations("BANANA"))))
    add("What is the remainder when 7^2026 is divided by 13?", pow(7, 2026, 13))
    add("How many points (x, y) with integer coordinates satisfy x^2 + y^2 <= 100?",
        sum(1 for x in range(-10, 11) for y in range(-10, 11) if x * x + y * y <= 100))
    add("What is the smallest positive integer n such that n! ends in exactly 25 zeros?",
        next(n for n in range(1, 1000) if _fact_zeros(n) == 25))
    add("What is the sum of all prime numbers less than 1000?", sum(_primes(999)))
    f = Fraction(sum(1 for a in range(1, 7) for b in range(1, 7) for c in range(1, 7)
                     if a + b + c == 10), 216)
    add("A fair six-sided die is rolled three times. The probability that the sum is 10 is m/n "
        "in lowest terms. What is m + n?", f.numerator + f.denominator)
    add("How many positive divisors does 7200 have?", _ndiv(7200))
    add("Three consecutive odd integers sum to 213. What is the product of the smallest and the "
        "largest?", 69 * 73)
    add("In how many ways can 10 identical candies be given to 4 children so that each child gets "
        "at least one candy?", math.comb(9, 3))
    add("How many integers n with 1 <= n <= 2026 satisfy gcd(n, 2026) = 1?",
        sum(1 for n in range(1, 2027) if math.gcd(n, 2026) == 1))
    add("With F(1) = F(2) = 1 and F(n) = F(n-1) + F(n-2), what is the remainder when F(100) is "
        "divided by 1000?", _fib(100) % 1000)
    add("What is the largest prime factor of 600851475143?", 6857)
    add("Two trains 300 km apart move toward each other at 70 km/h and 80 km/h. A bird flies at "
        "120 km/h back and forth between them until they meet. How many km does the bird fly?",
        300 / 150 * 120)
    add("A fair coin is flipped until two heads appear in a row. What is the expected number of "
        "flips?", 6)
    add("How many 4-digit numbers have digits that strictly increase from left to right?",
        sum(1 for n in range(1000, 10000) if all(a < b for a, b in zip(str(n), str(n)[1:]))))
    add("How many subsets of {1, 2, ..., 12} (including the empty set) have a sum divisible by 3?",
        sum(1 for m in range(1 << 12) if sum(i + 1 for i in range(12) if m >> i & 1) % 3 == 0))
    add("Real numbers x and y satisfy x + y = 17 and xy = 66. What is x^3 + y^3?",
        17 ** 3 - 3 * 66 * 17)
    add("How many squares of all sizes are there on an 8 by 8 chessboard grid?",
        sum(k * k for k in range(1, 9)))
    return items


# ---------------------------------------------------------------- hreason

def _jugs(a, b, goal):
    """The fewest fills, empties and pours to get goal liters in one jug."""
    seen, q = {(0, 0)}, deque([((0, 0), 0)])
    while q:
        (x, y), d = q.popleft()
        if goal in (x, y):
            return d
        p1, p2 = min(x, b - y), min(y, a - x)
        for s in ((a, y), (x, b), (0, y), (x, 0), (x - p1, y + p1), (x + p2, y - p2)):
            if s not in seen:
                seen.add(s)
                q.append((s, d + 1))


def _seating():
    """Five people in seats 1..5 (left to right): D at the left end, A not at
    an end, B immediately right of C, E somewhere right of A, C not next to D,
    E not next to B.
    The seat of E (unique: checked)."""
    sols = []
    for p in itertools.permutations("ABCDE"):
        pos = {c: i + 1 for i, c in enumerate(p)}
        if pos["D"] == 1 and pos["A"] not in (1, 5) and pos["B"] == pos["C"] + 1 and \
                pos["E"] > pos["A"] and abs(pos["C"] - pos["D"]) != 1 and abs(pos["E"] - pos["B"]) != 1:
            sols.append(pos)
    assert len(sols) == 1, sols
    return sols[0]["E"]


def word_grader(pattern, note):
    def g(msg):
        c = msg.get("content") or ""
        m = re.findall(pattern, c, re.I)
        ok = bool(m) and m[-1].lower() == note.lower()
        return (1.0 if ok else 0.0), "want %s, got %s" % (note, m[-1] if m else None)
    return g


def letter_last(want):
    def g(msg):
        m = re.findall(r"\b([ABCD])\b", (msg.get("content") or ""))
        got = m[-1] if m else None
        return (1.0 if got == want else 0.0), "want %s, got %s" % (want, got)
    return g


def _hreason():
    items = []

    def add(q, grade):
        items.append({"id": "hreason_%02d" % (len(items) + 1), "cat": "hreason", "messages": user(q),
                      "grade": grade})
    add("On an island, knights always tell the truth and knaves always lie. A says: \"B is a "
        "knave.\" B says: \"A and I are both knights.\" Which is true?\nA. Both are knights\nB. A "
        "is a knight, B is a knave\nC. A is a knave, B is a knight\nD. Both are knaves\nAnswer "
        "with only the letter on the last line.", letter_last("B"))
    add("What number comes next: 2, 6, 12, 20, 30, 42, ?" + " Give the number on the last line.",
        num_grader(56))
    seat = _seating()
    add("Five people A, B, C, D, E sit in a row of seats numbered 1 to 5 from left to right. D "
        "sits in seat 1. A is not at either end. B sits immediately to the right of C. E sits "
        "somewhere to the right of A. C does not sit next to D. E does not sit next to B. In which "
        "seat does E sit? Give the "
        "seat number on the last line.", num_grader(seat))
    day = datetime.date(2026, 12, 25).strftime("%A")
    add("2026-01-01 is a Thursday. What day of the week is 2026-12-25? Answer with the day name "
        "on the last line.", word_grader(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", day))
    add("You have a 5-liter jug and a 3-liter jug and unlimited water. One step is filling a jug, "
        "emptying a jug, or pouring one jug into the other until the first is empty or the second "
        "is full. What is the fewest number of steps to have exactly 4 liters in a jug? Give the "
        "number on the last line.", num_grader(_jugs(5, 3, 4)))
    add("A is twice as old as B was when A was as old as B is now. A is 24 years old. How old is "
        "B? Give the number on the last line.", num_grader(18))
    add("Behind one of 4 doors is a prize. You pick a door. The host, who knows where the prize "
        "is, opens 2 of the other doors, both empty, and offers you the last closed door. What is "
        "the probability in percent that you win if you switch? Give the number on the last line.",
        num_grader(75))
    add("A snail climbs 3 meters up a 20-meter wall each day and slides back 2 meters each night. "
        "On which day does it first reach the top? Give the number on the last line.",
        num_grader(18))
    add("Exactly one of these three statements is true. (1) The key is in the box. (2) The key is "
        "not in the bag. (3) The key is not in the box. The key is in exactly one of: the box, the "
        "bag, the drawer. Where is the key?\nA. the box\nB. the bag\nC. the drawer\nD. cannot be "
        "determined\nAnswer with only the letter on the last line.", letter_last("B"))
    add("A clock's hands overlap at 12:00. How many times do the hour and minute hands overlap in "
        "24 hours, counting 12:00 midnight at the start once and not the end? Give the number on "
        "the last line.", num_grader(22))
    return items


# ---------------------------------------------------------------- hcode

HCODE = [
    ("eval_expr", "Write a Python function eval_expr(s: str) -> float that evaluates an arithmetic "
     "expression with + - * /, parentheses, unary minus, decimal numbers and spaces, with the usual "
     "precedence and left associativity. Do not use eval or exec.",
     ["assert abs(eval_expr('2 + 3 * 4') - 14) < 1e-9", "assert abs(eval_expr('(2 + 3) * 4') - 20) < 1e-9",
      "assert abs(eval_expr('-3 + 10 / 4') - (-0.5)) < 1e-9", "assert abs(eval_expr('2 - 3 - 4') - (-5)) < 1e-9",
      "assert abs(eval_expr('-(1+2)*-(3)') - 9) < 1e-9", "assert abs(eval_expr('1.5*2') - 3) < 1e-9",
      "assert abs(eval_expr('8/4/2') - 1) < 1e-9"]),
    ("edit_distance", "Write a Python function edit_distance(a: str, b: str) -> int returning the "
     "Levenshtein distance (insert, delete, substitute, each cost 1).",
     ["assert edit_distance('kitten', 'sitting') == 3", "assert edit_distance('', 'abc') == 3",
      "assert edit_distance('flaw', 'lawn') == 2", "assert edit_distance('same', 'same') == 0",
      "assert edit_distance('intention', 'execution') == 5"]),
    ("lis_length", "Write a Python function lis_length(nums: list[int]) -> int returning the length "
     "of the longest strictly increasing subsequence, in O(n log n).",
     ["assert lis_length([10,9,2,5,3,7,101,18]) == 4", "assert lis_length([]) == 0",
      "assert lis_length([7,7,7]) == 1", "assert lis_length(list(range(5000))) == 5000",
      "assert lis_length([0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]) == 6"]),
    ("regex_match", "Write a Python function regex_match(s: str, p: str) -> bool where p may contain "
     "'.' (any single character) and '*' (zero or more of the preceding element), and the match "
     "must cover the whole string. Do not use the re module.",
     ["assert not regex_match('aa', 'a')", "assert regex_match('aa', 'a*')", "assert regex_match('ab', '.*')",
      "assert regex_match('aab', 'c*a*b')", "assert not regex_match('mississippi', 'mis*is*p*.')",
      "assert regex_match('', 'a*b*')", "assert not regex_match('ab', '.*c')"]),
    ("topo_sort", "Write a Python function topo_sort(n: int, edges: list[tuple[int, int]]) -> "
     "list[int] | None returning an order of the nodes 0..n-1 such that for every edge (u, v) u comes "
     "before v, or None if the graph has a cycle.",
     ["def _ok(n, e):\n    o = topo_sort(n, e)\n    if o is None: return None\n    p = {v: i for i, v in enumerate(o)}\n"
      "    return sorted(o) == list(range(n)) and all(p[u] < p[v] for u, v in e)",
      "assert _ok(4, [(0,1),(1,2),(0,3),(3,2)])", "assert topo_sort(3, [(0,1),(1,2),(2,0)]) is None",
      "assert _ok(1, [])", "assert _ok(5, [])", "assert topo_sort(2, [(1,1)]) is None"]),
    ("word_break", "Write a Python function word_break(s: str, words: list[str]) -> bool that says "
     "whether s can be split into a sequence of one or more words from the list (words may repeat).",
     ["assert word_break('leetcode', ['leet','code'])", "assert word_break('applepenapple', ['apple','pen'])",
      "assert not word_break('catsandog', ['cats','dog','sand','and','cat'])",
      "assert not word_break('a'*40+'b', ['a','aa','aaa'])"]),
    ("median_finder", "Write a Python class MedianFinder with add(self, x: float) and median(self) -> "
     "float, where median runs in O(1) and add in O(log n).",
     ["m = MedianFinder(); m.add(1); m.add(2); assert m.median() == 1.5; m.add(3); assert m.median() == 2",
      "m = MedianFinder()\nfor x in [5, 15, 1, 3]: m.add(x)\nassert m.median() == 4",
      "m = MedianFinder()\nfor x in range(1001): m.add(1000 - x)\nassert m.median() == 500"]),
    ("decode_ways", "Write a Python function decode_ways(s: str) -> int counting the ways to decode a "
     "digit string where 'A'..'Z' map to '1'..'26' (a '0' cannot stand alone or lead a two-digit code).",
     ["assert decode_ways('12') == 2", "assert decode_ways('226') == 3", "assert decode_ways('06') == 0",
      "assert decode_ways('10') == 1", "assert decode_ways('2101') == 1", "assert decode_ways('1111111111') == 89"]),
    ("n_queens", "Write a Python function n_queens(n: int) -> int returning the number of ways to place "
     "n non-attacking queens on an n by n board.",
     ["assert n_queens(1) == 1", "assert n_queens(4) == 2", "assert n_queens(6) == 4", "assert n_queens(8) == 92"]),
    ("knapsack", "Write a Python function knapsack(weights: list[int], values: list[int], cap: int) -> int "
     "returning the largest total value of items whose total weight is at most cap (each item at most "
     "once).",
     ["assert knapsack([1,3,4,5], [1,4,5,7], 7) == 9", "assert knapsack([], [], 10) == 0",
      "assert knapsack([5], [10], 4) == 0", "assert knapsack([10,20,30], [60,100,120], 50) == 220"]),
    ("spiral", "Write a Python function spiral(m: list[list[int]]) -> list[int] returning the elements "
     "of the matrix in clockwise spiral order starting at the top left.",
     ["assert spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]",
      "assert spiral([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) == [1,2,3,4,8,12,11,10,9,5,6,7]",
      "assert spiral([]) == []", "assert spiral([[1],[2],[3]]) == [1,2,3]"]),
    ("min_window", "Write a Python function min_window(s: str, t: str) -> str returning the shortest "
     "substring of s that contains every character of t with multiplicity, or '' if none; if several "
     "have the same length, return the leftmost.",
     ["assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'", "assert min_window('a', 'aa') == ''",
      "assert min_window('aa', 'aa') == 'aa'", "assert min_window('abc', '') == ''",
      "assert min_window('xyzzyx', 'xz') == 'xyz'", "assert min_window('cabwefgewcwaefgcf', 'cae') == 'cwae'"]),
    ("meeting_rooms", "Write a Python function min_rooms(meetings: list[tuple[int, int]]) -> int "
     "returning the fewest rooms needed for meetings given as (start, end) half-open intervals.",
     ["assert min_rooms([(0,30),(5,10),(15,20)]) == 2", "assert min_rooms([(7,10),(2,4)]) == 1",
      "assert min_rooms([]) == 0", "assert min_rooms([(1,5),(5,10)]) == 1", "assert min_rooms([(1,10)]*5) == 5"]),
    ("parse_duration", "Write a Python function parse_duration(s: str) -> int converting durations "
     "like '1h30m', '45s', '2d3h', '1h0m5s' (units d, h, m, s, each at most once, in that order) to "
     "seconds; raise ValueError for anything else, including an empty string or a unit out of order.",
     ["assert parse_duration('1h30m') == 5400", "assert parse_duration('45s') == 45",
      "assert parse_duration('2d3h') == 183600", "assert parse_duration('1h0m5s') == 3605",
      "import contextlib\nfor bad in ['', '5', '3m1h', '1x', 'h', '1h1h']:\n    try:\n        parse_duration(bad)\n"
      "        raise SystemExit('accepted ' + repr(bad))\n    except ValueError:\n        pass"]),
]

CODE_TAIL = (" Reply with the complete code in a single ```python code block; no tests, no example "
             "usage.")


def _hcode():
    items = []
    for name, q, tests in HCODE:
        def g(msg, tests=tests):
            ok, note = run_tests(extract_code(msg.get("content")), tests)
            return (1.0 if ok else 0.0), note
        items.append({"id": "hcode_" + name, "cat": "hcode", "messages": user(q + CODE_TAIL),
                      "grade": g})
    return items


# ---------------------------------------------------------------- htools

class Env:
    """A scripted world: files (path -> text), a log of what the model did,
    and handlers for the tools. Paths are normalized (no leading ./ or /home
    prefix differences)."""

    def __init__(self, files=None, extra=None):
        self.files = dict(files or {})
        self.log = []
        self.extra = extra or {}
        self.sent, self.events, self.written = [], [], {}

    @staticmethod
    def p(path):
        path = str(path or "").strip()
        while path.startswith("./"):
            path = path[2:]
        return path.lstrip("/").rstrip("/")

    def call(self, name, a):
        self.log.append((name, a))
        h = getattr(self, "t_" + name, None)
        if h is None:
            return "error: no tool %s" % name
        try:
            return h(**a)
        except TypeError as e:
            return "error: %s" % e

    def t_read_file(self, path):
        p = self.p(path)
        if p in self.files:
            return self.files[p]
        return "error: no such file: %s" % path

    def t_write_file(self, path, content):
        p = self.p(path)
        self.files[p] = content
        self.written[p] = content
        return "wrote %d bytes to %s" % (len(content), p)

    def t_search_files(self, pattern, directory="."):
        import fnmatch
        d = self.p(directory)
        d = "" if d in ("", ".") else d + "/"
        pat = pattern.split("/")[-1]
        out = [f for f in sorted(self.files) if f.startswith(d) and fnmatch.fnmatch(f.split("/")[-1], pat)]
        return json.dumps(out)

    def t_file_info(self, path):
        p = self.p(path)
        if p not in self.files:
            return "error: no such file: %s" % path
        return json.dumps({"path": p, "size_bytes": self.extra.get("sizes", {}).get(p, len(self.files[p]))})

    def t_run_command(self, command):
        """The command line of an agent: split at &&, ; and || (each part runs;
        the outputs follow each other) and at | (head and tail act on the text
        of the part before; other filters pass it on). cat, ls and pwd act on
        the files of the world (an agent tries them first: they had answered
        "command not found", and the Gemma 26B spent six of its twelve turns on
        them); the other commands are the scripted ones of the item
        (extra["commands"])."""
        outs = []
        for part in re.split(r"&&|\|\||;", str(command)):
            stages = [s.strip() for s in part.split("|") if s.strip()]
            if not stages:
                continue
            text = self._run_one(stages[0])
            for st in stages[1:]:
                text = self._pipe(st, text)
            outs.append(text)
        outs = [o for o in outs if o]
        return "\n".join(outs) if outs else "command not found or no output"

    def _run_one(self, cmd):
        import shlex
        try:
            words = shlex.split(cmd)
        except ValueError:
            words = cmd.split()
        words = [w for w in words if not re.match(r"^\d*[<>]", w)]      # 2>&1, >/dev/null
        if words and words[0] == "cat":
            out = []
            for f in [w for w in words[1:] if not w.startswith("-")]:
                p = self.p(f)
                out.append(self.files[p].rstrip("\n") if p in self.files
                           else "cat: %s: No such file or directory" % f)
            return "\n".join(out)
        if words and words[0] == "ls":
            args = [w for w in words[1:] if not w.startswith("-")] or ["."]
            out = []
            for a in args:
                d = self.p(a)
                if d in self.files:
                    out.append(d)
                    continue
                pre = "" if d in ("", ".") else d + "/"
                names = sorted({f[len(pre):].split("/")[0] + ("/" if "/" in f[len(pre):] else "")
                                for f in self.files if f.startswith(pre)})
                if not names:
                    out.append("ls: cannot access '%s': No such file or directory" % a)
                else:
                    out += names
            return "\n".join(out)
        if words and words[0] == "pwd":
            return "/home/user/project"
        for pat, out in self.extra.get("commands", []):
            if re.search(pat, cmd):
                return out(cmd) if callable(out) else out
        return ""

    @staticmethod
    def _pipe(stage, text):
        """head and tail (-n N, -N) on the lines of text; other filters keep it."""
        w = stage.split()
        if w and w[0] in ("head", "tail"):
            m = re.search(r"-n\s*(\d+)|-(\d+)", stage)
            n = int(m.group(1) or m.group(2)) if m else 10
            lines = text.splitlines()
            return "\n".join(lines[:n] if w[0] == "head" else lines[-n:])
        return text

    def t_get_weather(self, city, unit="celsius"):
        w = self.extra.get("weather", {}).get(city.strip().lower())
        if w is None:
            return "error: unknown city %s" % city
        t = w if unit == "celsius" else round(w * 9 / 5 + 32)
        return json.dumps({"city": city, "temperature": t, "unit": unit})

    def t_send_email(self, to, subject, body):
        self.sent.append({"to": to, "subject": subject, "body": body})
        return "sent"

    def t_list_events(self, date):
        return json.dumps(self.extra.get("events", {}).get(date.strip(), []))

    def t_create_event(self, title, date, time=None, duration_minutes=60):
        self.events.append({"title": title, "date": date, "time": time})
        return "created"

    def t_get_rate(self, base, quote):
        r = self.extra.get("rates", {}).get((base.upper(), quote.upper()))
        return json.dumps({"base": base, "quote": quote, "rate": r}) if r else "error: no rate"

    def t_rename_file(self, src, dst):
        s, d = self.p(src), self.p(dst)
        if s not in self.files:
            return "error: no such file: %s" % src
        self.files[d] = self.files.pop(s)
        return "renamed"


HT = {
    "read_file": fn("read_file", "Read a text file.", {"path": S}, ["path"]),
    "write_file": fn("write_file", "Write text to a file, replacing it.", {"path": S, "content": S}, ["path", "content"]),
    "search_files": fn("search_files", "List files whose names match a glob pattern (like *.log) under a directory, "
                       "recursively. Returns a JSON list of paths.", {"pattern": S, "directory": S}, ["pattern"]),
    "file_info": fn("file_info", "Size of a file in bytes.", {"path": S}, ["path"]),
    "run_command": fn("run_command", "Run a shell command; returns its output.", {"command": S}, ["command"]),
    "get_weather": fn("get_weather", "Current weather for a city.", {"city": S, "unit": {"type": "string",
                      "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    "send_email": fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    "list_events": fn("list_events", "The calendar events of a date (YYYY-MM-DD).", {"date": S}, ["date"]),
    "create_event": fn("create_event", "Create a calendar event.", {"title": S, "date": {"type": "string",
                       "description": "YYYY-MM-DD"}, "time": {"type": "string", "description": "HH:MM, 24h"},
                       "duration_minutes": {"type": "integer"}}, ["title", "date", "time"]),
    "get_rate": fn("get_rate", "The current exchange rate from one currency to another.", {"base": S, "quote": S},
                   ["base", "quote"]),
    "rename_file": fn("rename_file", "Rename or move a file.", {"src": S, "dst": S}, ["src", "dst"]),
}

SYS_T = ("You are an agent with tools. Today is 2026-03-10 (a Tuesday). Use the tools to do the task; "
         "do not ask the user questions. When you are done, give a short final answer.")


def htools_items():
    items = []

    def add(name, task, tools, make_env, grade):
        items.append({"id": "htools_" + name, "cat": "htools",
                      "messages": [{"role": "system", "content": SYS_T}] + user(task),
                      "tools": [HT[t] for t in tools], "env": make_env, "grade_env": grade})

    def largest_log():
        files = {"var/app/a.log": "x", "var/app/old/b.log": "x", "var/app/c.txt": "x", "var/app/d.log": "x"}
        return Env(files, {"sizes": {"var/app/a.log": 18234, "var/app/old/b.log": 912340,
                                     "var/app/d.log": 551, "var/app/c.txt": 9999999}})

    def g_largest(msg, env, trace):
        c = msg.get("content") or ""
        ok = "b.log" in c and re.search(r"912,?340", c)
        return (1.0 if ok else 0.0), "answer %r" % c[:100]
    add("largest_log", "Find the largest .log file under /var/app and tell me its name and size in bytes.",
        ["search_files", "file_info"], largest_log, g_largest)

    def config():
        return Env({"config.json": json.dumps({"host": "0.0.0.0", "port": 3000, "debug": False,
                                               "workers": 4}, indent=2)})

    def g_config(msg, env, trace):
        try:
            d = json.loads(env.files.get("config.json", ""))
        except Exception:
            return 0.0, "config.json not JSON"
        ok = d.get("port") == 8080 and d.get("host") == "0.0.0.0" and d.get("debug") is False and d.get("workers") == 4
        return (1.0 if ok else 0.0), "config now %s" % d
    add("edit_config", "In config.json, change the port to 8080 and keep everything else the same.",
        ["read_file", "write_file"], config, g_config)

    def office():
        return Env({"office.txt": "Head office: 14 Rue de la République, 69002 Lyon, France\nPhone: +33 4 00 00 00 00\n"},
                   {"weather": {"lyon": 9, "paris": 12}})

    def g_office(msg, env, trace):
        c = msg.get("content") or ""
        called = any(n == "get_weather" and "lyon" in str(a.get("city", "")).lower() for n, a in env.log)
        ok = called and re.search(r"\b9\b", c)
        return (1.0 if ok else 0.0), "weather call Lyon %s, answer %r" % (called, c[:80])
    add("office_weather", "What is the weather right now in the city where our office is? The address is in "
        "office.txt. Answer in celsius.", ["read_file", "get_weather"], office, g_office)

    def todos():
        return Env({"src/a.py": "x = 1  # TODO: rename\n# TODO check\ny = 2\n",
                    "src/pkg/b.py": "def f():\n    pass  # TODO\n",
                    "src/pkg/c.txt": "TODO TODO TODO\n",
                    "src/d.py": "print('no tasks here')\n"})

    def g_todos(msg, env, trace):
        # the number reported as the total (next to "total" or "TODO"), not
        # the last number of the answer (which may count the files)
        c = (msg.get("content") or "").replace("*", "")
        m = re.findall(r"(?i)(?:total[^0-9\n]{0,25}(\d+))|(?:(\d+)\s+TODO)", c)
        got = [int(a or b) for a, b in m]
        return (1.0 if got and set(got) == {3} else 0.0), "totals %s, answer %r" % (got, c[:80])
    add("count_todos", "Count the TODO comments in all the Python files under src/ (recursively) and tell me "
        "the total.", ["search_files", "read_file"], todos, g_todos)

    def report():
        return Env({"report.txt": "Q1 summary: revenue rose 12% to $4.2M; churn fell to 3.1%. The new "
                    "onboarding flow shipped on Feb 20. Risks: two key hires still open.\n"})

    def g_report(msg, env, trace):
        if not env.sent:
            return 0.0, "no email"
        m = env.sent[-1]
        ok = m["to"].strip().lower() == "alice@example.com" and "report" in m["subject"].lower() and \
            re.search(r"12\s?%|4\.2", m["body"])
        return (1.0 if ok else 0.0), "email %s" % json.dumps(m)[:120]
    add("email_summary", "Send alice@example.com an email with the subject 'Report' and a one-sentence summary "
        "of report.txt.", ["read_file", "send_email"], report, g_report)

    def tests_env():
        out = ("============================= test session starts ==============================\n"
               "collected 42 items\n\ntests/test_api.py ........................ [ 57%]\n"
               "tests/test_db.py ............F...... [100%]\n\n=================================== FAILURES "
               "===================================\n____________________ test_migration_rollback "
               "____________________\nAssertionError: expected 3 tables, found 2\n"
               "=========================== 1 failed, 41 passed in 3.2s ===========================\n")
        return Env({}, {"commands": [(r"pytest", out)]})

    def g_tests(msg, env, trace):
        c = msg.get("content") or ""
        ran = any(n == "run_command" and "pytest" in str(a.get("command", "")) for n, a in env.log)
        ok = ran and "test_migration_rollback" in c
        return (1.0 if ok else 0.0), "ran %s, answer %r" % (ran, c[:80])
    add("failing_test", "Run the test suite with pytest and tell me which test failed, if any.",
        ["run_command"], tests_env, g_tests)

    def cal():
        return Env({}, {"events": {"2026-03-11": [
            {"title": "Standup", "time": "09:00", "duration_minutes": 30},
            {"title": "Design review", "time": "09:30", "duration_minutes": 90},
            {"title": "Lunch", "time": "12:00", "duration_minutes": 60},
            {"title": "1:1", "time": "13:00", "duration_minutes": 30},
            {"title": "Focus", "time": "14:00", "duration_minutes": 120}]}})

    def g_cal(msg, env, trace):
        if not env.events:
            return 0.0, "no event"
        e = env.events[-1]
        ok = e["date"] == "2026-03-11" and str(e.get("time", "")).strip() in ("11:00",)
        return (1.0 if ok else 0.0), "event %s" % e
    add("schedule", "Book a 1-hour meeting titled 'Planning' tomorrow at the earliest time between 09:00 and "
        "17:00 when I am free. Check my calendar first.", ["list_events", "create_event"], cal, g_cal)

    def fx():
        return Env({}, {"rates": {("USD", "EUR"): 0.9174, ("EUR", "USD"): 1.09}})

    def g_fx(msg, env, trace):
        c = (msg.get("content") or "").replace(",", "")
        nums = [float(x) for x in re.findall(r"\d+\.\d+|\d+", c)]
        ok = any(abs(x - 250 * 0.9174) < 0.06 for x in nums)
        return (1.0 if ok else 0.0), "answer %r" % c[:80]
    add("currency", "How many euros is 250 US dollars at the current rate? Give the amount to two decimals.",
        ["get_rate"], fx, g_fx)

    def hosts():
        down = "db2.internal"

        def ping(cmd):
            h = re.findall(r"[\w.-]+\.internal", cmd)
            if not h:
                return "usage: ping host"
            if h[0] == down:
                return "PING %s: 3 packets transmitted, 0 received, 100%% packet loss" % h[0]
            return "PING %s: 3 packets transmitted, 3 received, 0%% packet loss" % h[0]
        return Env({"hosts.txt": "web1.internal\nweb2.internal\ndb1.internal\ndb2.internal\ncache.internal\n"},
                   {"commands": [(r"ping", ping)]})

    def g_hosts(msg, env, trace):
        c = msg.get("content") or ""
        pinged = {re.findall(r"[\w.-]+\.internal", str(a.get("command", "")))[0] for n, a in env.log
                  if n == "run_command" and re.findall(r"[\w.-]+\.internal", str(a.get("command", "")))}
        ok = "db2.internal" in c and "db2.internal" in pinged
        return (1.0 if ok else 0.0), "pinged %d, answer %r" % (len(pinged), c[:80])
    add("host_down", "Ping each host listed in hosts.txt (one ping command per host) and tell me which one is "
        "down.", ["read_file", "run_command"], hosts, g_hosts)

    def photos():
        return Env({"photos/a.jpeg": "1", "photos/b.jpeg": "2", "photos/c.png": "3", "photos/trip/d.jpeg": "4"})

    def g_photos(msg, env, trace):
        want = {"photos/a.jpg", "photos/b.jpg", "photos/c.png", "photos/trip/d.jpg"}
        got = set(env.files)
        return (1.0 if got == want else 0.0), "files %s" % sorted(got)
    add("rename_jpeg", "Rename every .jpeg file under photos/ (including subfolders) so it ends in .jpg instead.",
        ["search_files", "rename_file"], photos, g_photos)
    return items


# ---------------------------------------------------------------- hinstruct

def _sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def _hinstruct():
    def chk(f, note):
        def g(msg):
            try:
                ok = bool(f(msg.get("content") or ""))
            except Exception as e:
                return 0.0, repr(e)[:80]
            return (1.0 if ok else 0.0), note
        return g

    def jarr(t):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
        a = json.loads(t)
        return isinstance(a, list) and len(a) == 4 and all(
            isinstance(o, dict) and set(o) == {"name", "score"} and isinstance(o["name"], str) and
            isinstance(o["score"], int) and 0 <= o["score"] <= 100 for o in a) and \
            [o["score"] for o in a] == sorted([o["score"] for o in a], reverse=True)

    def table(t):
        rows = [l for l in t.strip().splitlines() if l.strip().startswith("|")]
        if len(rows) != 5:
            return False
        head = [c.strip().lower() for c in rows[0].strip("|").split("|")]
        return head == ["country", "capital"] and set(rows[1].replace("|", "").strip()) <= set("-: ")

    I = [
        ("lamp", "Write a product description of a desk lamp in exactly 3 sentences, each under 15 words, "
         "without using the word 'light' (or 'lights', 'lighting').",
         lambda t: len(_sents(t)) == 3 and all(len(words(s)) < 15 for s in _sents(t)) and
         not re.search(r"(?i)\blight", t), "3 sents <15w no light"),
        ("json_array", "Return only a JSON array of 4 objects, each with exactly the keys \"name\" (a string) "
         "and \"score\" (an integer from 0 to 100), sorted by score from highest to lowest.", jarr, "JSON array"),
        ("acrostic", "Write 5 lines of verse about winter where the lines start with the letters A, B, C, D, E "
         "in that order. Only the 5 lines.",
         lambda t: [l.strip().lstrip("*_\"'")[:1].upper() for l in t.strip().splitlines() if l.strip()] ==
         list("ABCDE"), "acrostic"),
        ("french", "Réponds en français, en exactement deux phrases, et mentionne 'Paris' exactement une fois : "
         "pourquoi la tour Eiffel est-elle célèbre ?",
         lambda t: len(_sents(t)) == 2 and len(re.findall(r"Paris", t)) == 1 and
         not re.search(r"(?i)\b(the|is|and|famous)\b", t), "2 French sents, Paris once"),
        ("sorted_list", "List 6 programming languages as a numbered list (1. to 6.), in alphabetical order, with "
         "no explanations.",
         lambda t: (lambda L: len(L) == 6 and L == sorted(L, key=str.lower))(
             [re.sub(r"^\s*\d+[.)]\s*", "", l).strip("* ") for l in t.splitlines() if re.match(r"^\s*\d+[.)]", l)]),
         "6 sorted"),
        ("seven_words", "Write a 4-line poem about the moon where every line has exactly 7 words. Only the poem.",
         lambda t: (lambda L: len(L) == 4 and all(len(words(l)) == 7 for l in L))(
             [l for l in t.strip().splitlines() if l.strip()]), "4x7 words"),
        ("table", "Give a markdown table with the columns Country and Capital and exactly 3 data rows, and no "
         "other text.", table, "table 3 rows"),
        ("banana", "Give three lines: the word 'banana' written backwards; then 'banana' in uppercase; then the "
         "number of letters in 'banana'. Nothing else.",
         lambda t: [l.strip().strip("`*") for l in t.strip().splitlines() if l.strip()] == ["ananab", "BANANA", "6"],
         "3 exact lines"),
        ("py_dict", "Output only a Python dict literal with the keys 'a' (an int), 'b' (a list of 3 strings) "
         "and 'c' (None).",
         lambda t: (lambda d: isinstance(d, dict) and set(d) == {"a", "b", "c"} and isinstance(d["a"], int) and
                    isinstance(d["b"], list) and len(d["b"]) == 3 and all(isinstance(x, str) for x in d["b"]) and
                    d["c"] is None)(ast.literal_eval(re.sub(r"^```(?:python)?\s*|\s*```$", "", t.strip()))), "dict"),
        ("no_comma", "Explain how vaccines work in 3 to 5 sentences without using any commas and without the "
         "word 'virus'.",
         lambda t: 3 <= len(_sents(t)) <= 5 and "," not in t and not re.search(r"(?i)\bvirus", t), "no comma"),
    ]
    return [{"id": "hinstruct_" + n, "cat": "hinstruct", "messages": user(q), "grade": chk(f, note)}
            for n, q, f, note in I]


# ---------------------------------------------------------------- hneedle

def hneedle_item(n_tokens, seed):
    import random
    from suite import FILLER
    r = random.Random(seed)
    codes = {c: r.randint(1000, 9999) for c in ("red", "green", "blue")}
    target = int(n_tokens * 4.4)
    out, size = [], 0
    while size < target:
        s = r.choice(FILLER)
        out.append(s)
        size += len(s) + 1
    for c, d in zip(("red", "blue", "green"), (0.2, 0.5, 0.8)):
        out.insert(int(len(out) * d), "Note for the guards: the code for the %s door is %d." % (c, codes[c]))
    q = " ".join(out) + ("\n\nWhat is the sum of the codes of the red door and the green door? Give the number "
                         "on the last line.")
    want = codes["red"] + codes["green"]
    return {"id": "hneedle_%dk" % (n_tokens // 1000), "cat": "hneedle", "messages": user(q),
            "grade": num_grader(want), "max_tokens": 4000}


def hneedle(lengths=(32000, 128000)):
    return [hneedle_item(n, n) for n in lengths]


def hard_items(lengths=(32000, 128000)):
    return _hmath() + _hreason() + _hcode() + htools_items() + _hinstruct() + hneedle(lengths)
