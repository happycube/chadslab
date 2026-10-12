"""The expert tier of the quality suite (scripts/quality_suite.py --tier expert).

The hard tier (hard.py) reached its ceiling on Qwen3.8 (66 of 66). These
items are harder still, so that a strong model misses some and a small loss
of precision moves the count:

    xmath    counting and number theory with careful casework; every key from
             brute force or an exact DP here
    xreason  a Zebra puzzle and other puzzles, unique answers checked here
    xcode    whole programs (a JSON parser, a regex engine, a Lisp, a sudoku
             solver, ...), tested against hidden references or brute force on
             random inputs
    xtools   long agent tasks with planning and errors to recover from
    xneedle  two-hop retrieval among near-duplicate records, and counting over
             the whole text (--needle-expert lengths, 64K and 128K), and the
             count alone at 256K (--needle-expert-count; near the 262144 context)
"""
from __future__ import annotations

import itertools
import json
import math
import re
from collections import deque
from fractions import Fraction
from functools import lru_cache

from suite import FILLER, extract_code, fn, num_grader, run_tests, user, S
from hard import Env, letter_last, word_grader

TAIL = (" This is a hard problem: work carefully and check your work. Give the final answer as a "
        "single number on the last line.")


# ---------------------------------------------------------------- xmath

def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _tau(n):
    t, d = 1, 2
    while d * d <= n:
        e = 0
        while n % d == 0:
            n //= d
            e += 1
        t *= e + 1
        d += 1
    return t * (2 if n > 1 else 1)


def _paths_below(n, k):
    """Monotone paths (0,0) -> (n,n), right/up, never with y > x + k."""
    f = [[0] * (n + 1) for _ in range(n + 1)]
    for x in range(n + 1):
        for y in range(n + 1):
            if y > x + k:
                continue
            if x == 0 and y == 0:
                f[x][y] = 1
                continue
            f[x][y] = (f[x - 1][y] if x else 0) + (f[x][y - 1] if y else 0)
    return f[n][n]


def _domino_3xn(n):
    """Tilings of a 3 x n board by 1 x 2 dominoes (broken-profile DP)."""
    @lru_cache(None)
    def go(col, mask):
        if col == n:
            return 1 if mask == 0 else 0

        def fill(row, cur, nxt):
            if row == 3:
                return go(col + 1, nxt)
            if cur >> row & 1:
                return fill(row + 1, cur, nxt)
            t = fill(row + 1, cur | 1 << row, nxt | 1 << row)          # horizontal
            if row < 2 and not cur >> (row + 1) & 1:
                t += fill(row + 2, cur | 3 << row, nxt)                # vertical
            return t
        return fill(0, mask, 0)
    return go(0, 0)


def _digit_sum_count(ndig, s):
    f = [1] + [0] * s
    for _ in range(ndig):
        g = [0] * (s + 1)
        for v in range(s + 1):
            if f[v]:
                for d in range(10):
                    if v + d <= s:
                        g[v + d] += f[v]
        f = g
    return f[s]


def _xmath():
    items = []

    def add(q, ans):
        items.append({"id": "xmath_%02d" % (len(items) + 1), "cat": "xmath", "messages": user(q + TAIL),
                      "grade": num_grader(ans)})
    n = 7200
    add("How many ordered triples (a, b, c) of positive integers satisfy a*b*c = 7200 and a <= b <= c?",
        sum(1 for a in _divisors(n) for b in _divisors(n // a) if a <= b and b <= n // a // b and
            n % (a * b) == 0))
    add("How many binary strings of length 15 contain no three consecutive 1s?",
        sum(1 for m in range(1 << 15) if "111" not in format(m, "015b")))
    add("How many lattice paths from (0, 0) to (10, 10) using unit steps right or up never pass "
        "through a point (x, y) with y > x + 2?", _paths_below(10, 2))
    add("How many integers from 1 to 999999 have a digit sum of exactly 27?", _digit_sum_count(6, 27))
    add("What are the last three digits of 3^(3^(3^3))? Give them as an integer (for example 7 for 007).",
        pow(3, 3 ** 27, 1000))
    add("In how many ways can a 3 by 12 rectangle be tiled with 1 by 2 dominoes?", _domino_3xn(12))
    add("How many permutations p of 1, 2, ..., 8 satisfy p(i) != i and p(i) != i + 1 for every i?",
        sum(1 for p in itertools.permutations(range(1, 9))
            if all(p[i] != i + 1 and p[i] != i + 2 for i in range(8))))
    add("What is the smallest positive integer that has exactly 60 positive divisors?",
        next(m for m in itertools.count(1) if _tau(m) == 60))
    add("How many subsets of {1, 2, ..., 18} contain no two consecutive integers and have a sum "
        "divisible by 6? (The empty set counts.)",
        sum(1 for m in range(1 << 18) if not m & (m >> 1) and
            sum(i + 1 for i in range(18) if m >> i & 1) % 6 == 0))
    add("How many 4 by 4 matrices with entries 0 or 1 have exactly two 1s in every row and every column?",
        sum(1 for rows in itertools.product([r for r in itertools.product((0, 1), repeat=4) if sum(r) == 2],
                                            repeat=4) if all(sum(c) == 2 for c in zip(*rows))))
    add("How many positive integers n <= 10000 are such that both n and n + 1 have exactly 4 positive "
        "divisors?", sum(1 for k in range(1, 10001) if _tau(k) == 4 and _tau(k + 1) == 4))
    N2 = 2026 ** 2
    add("How many ordered pairs (x, y) of positive integers satisfy x^2 - y^2 = 2026^2?",
        sum(1 for d in range(1, 2026) if N2 % d == 0 and (d + N2 // d) % 2 == 0))
    a, b, cnt = 1, 1, 0
    for i in range(1, 2027):
        if a % 7 == 0:
            cnt += 1
        a, b = b, a + b
    add("With F(1) = F(2) = 1 and F(n) = F(n-1) + F(n-2), how many of F(1), F(2), ..., F(2026) are "
        "divisible by 7?", cnt)
    e = Fraction(6) * (1 - Fraction(5, 6) ** 6)
    add("A fair six-sided die is rolled 6 times. The expected number of distinct values that appear is "
        "m/n in lowest terms. What is m + n?", e.numerator + e.denominator)
    add("In how many ways can 8 rooks be placed on an 8 by 8 chessboard so that no two attack each other "
        "and no rook is on the main diagonal (the squares (i, i))?",
        sum(1 for p in itertools.permutations(range(8)) if all(p[i] != i for i in range(8))))
    add("In how many ways can 20 be written as an ordered sum of parts each equal to 1, 2 or 3?",
        (lambda f: f(f, 20))(lambda g, k: 1 if k == 0 else sum(g(g, k - j) for j in (1, 2, 3) if k >= j)))
    pw = {1}
    for base in range(2, 317):
        v = base * base
        while v <= 100000:
            pw.add(v)
            v *= base
    add("How many integers from 1 to 100000 inclusive are perfect powers m^k with integers m >= 1 and "
        "k >= 2 (1 counts)?", len(pw))
    add("How many triangles with integer side lengths (up to congruence) have perimeter 100?",
        sum(1 for a in range(1, 100) for b in range(a, 100) for c in [100 - a - b]
            if c >= b and a + b > c))
    return items


# ---------------------------------------------------------------- xreason

ZEBRA = {
    "nation": ["Norwegian", "Dane", "Brit", "German"],
    "color": ["yellow", "blue", "red", "green"],
    "drink": ["water", "tea", "milk", "coffee"],
    "pet": ["cat", "horse", "bird", "fish"],
}
ZEBRA_CLUES = [
    "The Norwegian lives in the first house.",
    "The Brit lives in the red house.",
    "The green house is immediately to the right of the red house.",
    "The German drinks coffee.",
    "Milk is drunk in the third house.",
    "The Dane keeps a horse.",
    "The cat lives in a house next to the blue house.",
    "The owner of the yellow house drinks water.",
    "The bird is kept in the red house.",
    "The tea drinker lives next to the Norwegian.",
]


def _zebra_solutions():
    sols = []
    P = list(itertools.permutations(range(4)))
    for nat in P:
        N = dict(zip(ZEBRA["nation"], nat))
        if N["Norwegian"] != 0:
            continue
        for col in P:
            C = dict(zip(ZEBRA["color"], col))
            if C["red"] != N["Brit"] or C["green"] != C["red"] + 1:
                continue
            for dr in P:
                D = dict(zip(ZEBRA["drink"], dr))
                if D["coffee"] != N["German"] or D["milk"] != 2 or D["water"] != C["yellow"] or \
                        abs(D["tea"] - N["Norwegian"]) != 1:
                    continue
                for pe in P:
                    Q = dict(zip(ZEBRA["pet"], pe))
                    if Q["horse"] != N["Dane"] or abs(Q["cat"] - C["blue"]) != 1 or Q["bird"] != C["red"]:
                        continue
                    sols.append({"nation": N, "pet": Q})
    return sols


def _mc(m=3, c=3, boat=2):
    start, goal = (m, c, 1), (0, 0, 0)
    seen, q = {start}, deque([(start, 0)])
    moves = [(a, b) for a in range(boat + 1) for b in range(boat + 1) if 1 <= a + b <= boat]
    while q:
        (x, y, s), d = q.popleft()
        if (x, y, s) == goal:
            return d
        for a, b in moves:
            nx, ny = (x - a, y - b) if s else (x + a, y + b)
            if not (0 <= nx <= m and 0 <= ny <= c):
                continue
            if (nx and ny > nx) or (m - nx and c - ny > m - nx):
                continue
            st = (nx, ny, 1 - s)
            if st not in seen:
                seen.add(st)
                q.append((st, d + 1))


def _liars():
    """Five people; person k says 'exactly k of us five are lying'. The number
    of truth-tellers (unique: checked)."""
    sols = []
    for truth in itertools.product((0, 1), repeat=5):
        lying = 5 - sum(truth)
        if all(bool(t) == (lying == k + 1) for k, t in enumerate(truth)):
            sols.append(sum(truth))
    assert len(sols) == 1, sols
    return sols[0]


def _xreason():
    items = []

    def add(q, g):
        items.append({"id": "xreason_%02d" % (len(items) + 1), "cat": "xreason", "messages": user(q),
                      "grade": g})
    sols = _zebra_solutions()
    assert len(sols) == 1, len(sols)
    owner = [n for n, h in sols[0]["nation"].items() if h == sols[0]["pet"]["fish"]][0]
    add("Four houses stand in a row, numbered 1 to 4 from left to right. Each has a different color "
        "(yellow, blue, red, green), an owner of a different nationality (Norwegian, Dane, Brit, German), "
        "a different drink (water, tea, milk, coffee) and a different pet (cat, horse, bird, fish).\n" +
        "\n".join("- " + c for c in ZEBRA_CLUES) + "\nWho keeps the fish? Answer with the nationality on "
        "the last line.", word_grader(r"(Norwegian|Dane|Brit|German)", owner))
    add("Three missionaries and three cannibals must cross a river in a boat that holds at most two people "
        "and cannot cross empty. On neither bank may the cannibals ever outnumber the missionaries when at "
        "least one missionary is there (people in the boat count as being on the bank the boat is at, after "
        "it lands). What is the fewest number of crossings? Give the number on the last line.",
        num_grader(_mc()))
    add("Five people each make one statement. Person 1 says: 'Exactly one of us five is lying.' Person 2 "
        "says: 'Exactly two of us are lying.' Person 3: 'Exactly three.' Person 4: 'Exactly four.' Person 5: "
        "'Exactly five.' Each person either always lies or always tells the truth. How many of them are "
        "telling the truth? Give the number on the last line.", num_grader(_liars()))
    t = Fraction(270, Fraction(11, 2))
    add("Between 3:00 and 4:00, the hour and minute hands of a clock point in exactly opposite directions "
        "once. At that moment, the number of minutes past 3:00 is m/n in lowest terms. What is m + n? Give "
        "the number on the last line.", num_grader(t.numerator + t.denominator))
    e = Fraction(0)
    reds, blues = 3, 5
    for order in set(itertools.permutations("R" * reds + "B" * blues)):
        e += Fraction(order.index("R") + 1)
    e /= math.comb(reds + blues, reds)
    add("A bag holds 3 red and 5 blue balls. Balls are drawn one at a time without replacement until the "
        "first red ball is drawn. The expected number of draws is m/n in lowest terms. What is m + n? Give "
        "the number on the last line.", num_grader(e.numerator + e.denominator))
    paths = math.comb(12, 6) - math.comb(6, 3) ** 2
    add("A city has a 6 by 6 grid of blocks, so 7 by 7 intersections, numbered (0, 0) to (6, 6). How many "
        "shortest routes go from (0, 0) to (6, 6) along the streets without passing through the "
        "intersection (3, 3)? Give the number on the last line.", num_grader(paths))
    surv = list(range(1, 42))
    i = 0
    while len(surv) > 1:
        i = (i + 2) % len(surv)
        surv.pop(i)
    add("41 people stand in a circle, numbered 1 to 41. Counting starts at person 1; every third person "
        "(3, 6, 9, ...) is removed, and counting continues around the circle from the next person until one "
        "remains. What is the number of the last person? Give the number on the last line.",
        num_grader(surv[0]))
    return items


def expert_items_part1():
    return _xmath() + _xreason()


# ---------------------------------------------------------------- xcode

XCODE = [
    ("json_parse", "Write a Python function json_parse(s: str) that parses a JSON text into Python values "
     "(dict, list, str, int, float, True, False, None), following the JSON grammar exactly: objects, "
     "arrays, strings with all escapes including \\uXXXX, numbers with sign, fraction and exponent (no "
     "leading zeros), whitespace. Raise ValueError for any invalid input. Do not use the json module, eval, "
     "or ast.",
     ["import json as _j",
      "_ok = ['{\"a\": [1, 2.5, -3e2, true, false, null], \"b\": {\"c\": \"x\\\\ny\\\\u00e9\\\\\"q\"}}', '[]', '{}',"
      " '\"\\\\\\\\\"', '0', '-0.5E+3', '[{\"k\": []}, [[]]]', ' { \"s\" : \"tab\\\\tend\" } ', '1e5', '\"\\\\u20ac\"',"
      " '[' * 40 + ']' * 40, '{\"x\":{\"y\":{\"z\":[0.125,-7,\"\"]}}}']",
      "for t in _ok:\n    assert json_parse(t) == _j.loads(t), t",
      "for t in ['{', '[1,]', '{\"a\" 1}', \"'x'\", '01', 'tru', '[1 2]', '\"\\\\x\"', '{\"a\":1,}', '', '[\"a]', '-', '1.', '.5', '{\"a\":}']:\n"
      "    try:\n        json_parse(t)\n        raise SystemExit('accepted ' + repr(t))\n    except ValueError:\n        pass"]),
    ("sudoku", "Write a Python function solve(grid: list[str]) -> list[str] that solves a 9x9 sudoku given as 9 "
     "strings of 9 characters, digits 1-9 or '.' for empty, and returns the solved grid in the same form. "
     "It must solve the hardest known puzzles in a few seconds.",
     ["g = ['8........', '..36.....', '.7..9.2..', '.5...7...', '....457..', '...1...3.', '..1....68', "
      "'..85...1.', '.9....4..']",
      "s = solve(g)",
      "assert len(s) == 9 and all(len(r) == 9 for r in s)",
      "assert all(g[i][j] in '.' + s[i][j] for i in range(9) for j in range(9))",
      "assert all(sorted(r) == list('123456789') for r in s)",
      "assert all(sorted(s[i][j] for i in range(9)) == list('123456789') for j in range(9))",
      "assert all(sorted(s[i][j] for i in range(b // 3 * 3, b // 3 * 3 + 3) for j in range(b % 3 * 3, b % 3 * 3 + 3)) "
      "== list('123456789') for b in range(9))"]),
    ("lisp", "Write a Python function eval_lisp(src: str) that evaluates a program in a small Scheme and returns "
     "the value of its last expression. It must support integers, #t and #f, symbols, (define name expr), "
     "(define (f args...) body), (lambda (args...) body), (if test then else), the operators + - * < > = "
     "(with + and * taking any number of arguments), recursion, and closures.",
     ["assert eval_lisp('(define (fact n) (if (< n 2) 1 (* n (fact (- n 1))))) (fact 10)') == 3628800",
      "assert eval_lisp('(define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))) (fib 15)') == 610",
      "assert eval_lisp('(define (make-adder n) (lambda (x) (+ x n))) (define add5 (make-adder 5)) (add5 10)') == 15",
      "assert eval_lisp('((lambda (a b) (- a b)) 10 3)') == 7",
      "assert eval_lisp('(if (= 1 2) 10 (* 2 3 4))') == 24",
      "assert eval_lisp('(define x 4) (define (sq y) (* y y)) (+ (sq x) (sq (+ x 1)))') == 41",
      "assert eval_lisp('(define (compose f g) (lambda (x) (f (g x)))) ((compose (lambda (x) (* x 2)) (lambda (x) (+ x 3))) 4)') == 14",
      "assert eval_lisp('(if #f 1 2)') == 2"]),
    ("skyline", "Write a Python function skyline(buildings: list[list[int]]) -> list[list[int]] returning the "
     "key points [x, height] of the skyline of buildings given as [left, right, height] (left < right, "
     "height > 0), sorted by x; a key point is where the height of the outline changes; the last has height "
     "0. Consecutive key points must have different heights.",
     ["import random\ndef _ref(b):\n    xs = sorted({x for l, r, h in b for x in (l, r)}); out = []; prev = 0\n"
      "    for x in xs:\n        h = max([hh for l, r, hh in b if l <= x < r], default=0)\n"
      "        if h != prev: out.append([x, h]); prev = h\n    return out",
      "assert skyline([]) == []",
      "assert [list(p) for p in skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]])] == "
      "[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]",
      "r = random.Random(5)\nfor _ in range(300):\n    b = [[l, l + r.randint(1, 6), r.randint(1, 5)] for l in "
      "[r.randint(0, 15) for _ in range(r.randint(1, 7))]]\n    assert [list(p) for p in skyline(b)] == _ref(b), b"]),
    ("pal_subseq", "Write a Python function count_pal_subseq(s: str) -> int returning the number of distinct "
     "non-empty palindromic subsequences of s (a string over 'a', 'b', 'c', 'd'), modulo 1_000_000_007. It "
     "must handle strings of length 1000 in under a second or two.",
     ["import random, itertools\ndef _brute(s):\n    seen = set()\n    for m in range(1, 1 << len(s)):\n"
      "        t = ''.join(s[i] for i in range(len(s)) if m >> i & 1)\n        if t == t[::-1]: seen.add(t)\n    return len(seen)",
      "r = random.Random(7)\nfor _ in range(120):\n    s = ''.join(r.choice('abcd') for _ in range(r.randint(1, 12)))\n"
      "    assert count_pal_subseq(s) == _brute(s), s",
      "assert count_pal_subseq('bccb') == 6",
      "import time\nt0 = time.time(); v = count_pal_subseq(''.join(random.Random(1).choice('abcd') for _ in range(1000)))\n"
      "assert isinstance(v, int) and 0 <= v < 1_000_000_007 and time.time() - t0 < 8"]),
    ("grid_k", "Write a Python function shortest_k(grid: list[list[int]], k: int) -> int returning the fewest "
     "steps (up, down, left, right) from the top-left to the bottom-right cell of a grid of 0 (free) and 1 "
     "(wall), where you may pass through at most k walls; return -1 if impossible. The start and end cells "
     "are 0.",
     ["import random\nfrom collections import deque\ndef _ref(g, k):\n    m, n = len(g), len(g[0]); q = deque([(0, 0, k, 0)]); seen = {(0, 0, k)}\n"
      "    while q:\n        i, j, r, d = q.popleft()\n        if (i, j) == (m - 1, n - 1): return d\n"
      "        for a, b in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):\n            if 0 <= a < m and 0 <= b < n:\n"
      "                rr = r - g[a][b]\n                if rr >= 0 and (a, b, rr) not in seen: seen.add((a, b, rr)); q.append((a, b, rr, d + 1))\n"
      "    return -1",
      "r = random.Random(11)\nfor _ in range(300):\n    m, n = r.randint(1, 7), r.randint(1, 7)\n"
      "    g = [[1 if r.random() < 0.4 else 0 for _ in range(n)] for _ in range(m)]; g[0][0] = g[-1][-1] = 0\n"
      "    k = r.randint(0, 3)\n    assert shortest_k([row[:] for row in g], k) == _ref(g, k), (g, k)"]),
    ("trap_2d", "Write a Python function trap_2d(h: list[list[int]]) -> int returning how much water a 2D "
     "elevation map traps after rain (water flows off the edges).",
     ["import random, heapq\ndef _ref(h):\n    if not h or not h[0]: return 0\n    m, n = len(h), len(h[0]); seen = [[False]*n for _ in range(m)]; q = []\n"
      "    for i in range(m):\n        for j in range(n):\n            if i in (0, m-1) or j in (0, n-1): heapq.heappush(q, (h[i][j], i, j)); seen[i][j] = True\n"
      "    w = 0\n    while q:\n        v, i, j = heapq.heappop(q)\n        for a, b in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):\n"
      "            if 0 <= a < m and 0 <= b < n and not seen[a][b]:\n                seen[a][b] = True; w += max(0, v - h[a][b])\n"
      "                heapq.heappush(q, (max(v, h[a][b]), a, b))\n    return w",
      "assert trap_2d([[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]) == 4",
      "r = random.Random(3)\nfor _ in range(300):\n    m, n = r.randint(1, 6), r.randint(1, 6)\n"
      "    h = [[r.randint(0, 6) for _ in range(n)] for _ in range(m)]\n    assert trap_2d([row[:] for row in h]) == _ref(h), h"]),
    ("alien_order", "Write a Python function alien_order(words: list[str]) -> str that, given words sorted in an "
     "unknown alphabet's order, returns a string of all the distinct letters in an order consistent with the "
     "sorting, or '' if no order is consistent (including when a word comes before its own proper prefix).",
     ["def _check(words):\n    o = alien_order(words)\n    letters = set(''.join(words))\n    cons = []\n    valid = True\n"
      "    for a, b in zip(words, words[1:]):\n        d = next(((x, y) for x, y in zip(a, b) if x != y), None)\n"
      "        if d is None:\n            if len(a) > len(b): valid = False\n        else: cons.append(d)\n"
      "    if valid:\n        import itertools\n        g = {c: set() for c in letters}\n        for x, y in cons: g[x].add(y)\n"
      "        state = {}\n        def cyc(u):\n            state[u] = 1\n            for v in g[u]:\n"
      "                if state.get(v) == 1 or (v not in state and cyc(v)): return True\n            state[u] = 2\n            return False\n"
      "        valid = not any(c not in state and cyc(c) for c in letters)\n"
      "    if not valid: return o == ''\n    pos = {c: i for i, c in enumerate(o)}\n"
      "    return sorted(o) == sorted(letters) and all(pos[x] < pos[y] for x, y in cons)",
      "assert _check(['wrt','wrf','er','ett','rftt'])", "assert _check(['z','x'])", "assert _check(['z','x','z'])",
      "assert _check(['abc','ab'])", "assert _check(['a'])", "assert _check(['ab','adc'])",
      "import random\nr = random.Random(9)\nfor _ in range(200):\n    alpha = list('abcdef'); r.shuffle(alpha); rank = {c: i for i, c in enumerate(alpha)}\n"
      "    ws = sorted({''.join(r.choice(alpha[:r.randint(2, 6)]) for _ in range(r.randint(1, 4))) for _ in range(r.randint(1, 8))}, key=lambda w: [rank[c] for c in w])\n"
      "    if r.random() < 0.3 and len(ws) > 1: ws[0], ws[-1] = ws[-1], ws[0]\n    assert _check(ws), ws"]),
    ("median_two", "Write a Python function median_two(a: list[int], b: list[int]) -> float returning the "
     "median of the union of two sorted lists (at least one non-empty) in O(log(min(m, n))) time.",
     ["import random, statistics, time",
      "r = random.Random(4)\nfor _ in range(500):\n    a = sorted(r.randint(-50, 50) for _ in range(r.randint(0, 9)))\n"
      "    b = sorted(r.randint(-50, 50) for _ in range(r.randint(0 if a else 1, 9)))\n"
      "    assert abs(median_two(a, b) - statistics.median(a + b)) < 1e-9, (a, b)",
      "a = list(range(0, 2_000_000, 2)); b = list(range(1, 2_000_000, 2))\nt0 = time.time()\n"
      "for _ in range(2000): median_two(a, b)\nassert time.time() - t0 < 4"]),
    ("regex_engine", "Write a Python function full_match(s: str, p: str) -> bool that says whether the whole "
     "string s matches the pattern p, where p uses literal letters, '.' (any one character), the postfix "
     "operators '*' (zero or more), '+' (one or more) and '?' (zero or one) applied to the preceding letter, "
     "'.' or parenthesized group, '|' (alternation, lowest precedence) and parentheses for grouping. Do not "
     "use the re module.",
     ["import random, re\ndef _gen(r, d):\n    if d <= 0 or r.random() < 0.3:\n        a = r.choice('ab.')\n    else:\n"
      "        k = r.random()\n        if k < 0.35: a = _gen(r, d - 1) + _gen(r, d - 1)\n"
      "        elif k < 0.6: a = '(' + _gen(r, d - 1) + '|' + _gen(r, d - 1) + ')'\n        else: a = '(' + _gen(r, d - 1) + ')'\n"
      "    if r.random() < 0.35: a = (a if len(a) == 1 or (a[0] == '(' and a[-1] == ')') else '(' + a + ')') + r.choice('*+?')\n    return a",
      "r = random.Random(13)\nfor _ in range(400):\n    p = _gen(r, 3)\n    for _ in range(6):\n"
      "        s = ''.join(r.choice('ab') for _ in range(r.randint(0, 7)))\n"
      "        assert full_match(s, p) == bool(re.fullmatch(p, s)), (s, p)",
      "assert full_match('abab', '(ab)+') and not full_match('aba', '(ab)+') and full_match('', '(a|b)*')",
      "assert full_match('ac', 'a(b|c)') and not full_match('a', 'a(b|c)')"]),
]

XCODE_TAIL = (" Reply with the complete code in a single ```python code block; no tests and no example usage.")


def _xcode():
    items = []
    for name, q, tests in XCODE:
        def g(msg, tests=tests):
            ok, note = run_tests(extract_code(msg.get("content")), tests, timeout=30)
            return (1.0 if ok else 0.0), note
        items.append({"id": "xcode_" + name, "cat": "xcode", "messages": user(q + XCODE_TAIL), "grade": g})
    return items


# ---------------------------------------------------------------- xtools

class XEnv(Env):
    def t_delete_file(self, path):
        p = self.p(path)
        if p not in self.files:
            return "error: no such file: %s" % path
        del self.files[p]
        self.extra.setdefault("deleted", []).append(p)
        return "deleted %s" % p

    def t_file_info(self, path):
        p = self.p(path)
        if p not in self.files:
            return "error: no such file: %s" % path
        return json.dumps({"path": p, "size_bytes": len(self.files[p]),
                           "modified": self.extra.get("mtime", {}).get(p, "2026-03-01")})

    def t_list_calendar(self, person, date):
        return json.dumps(self.extra.get("cal", {}).get(person.strip().lower(), {}).get(date.strip(), []))

    def t_create_event(self, title, date, time=None, attendees=None, duration_minutes=60):
        self.events.append({"title": title, "date": date, "time": time, "attendees": attendees})
        return "created"

    def t_get_stock(self, symbol):
        sym = symbol.strip().upper()
        n = self.extra.setdefault("asks", {})
        n[sym] = n.get(sym, 0) + 1
        if n[sym] == 1:
            return "error: rate limited, retry the same request"
        p = self.extra.get("prices", {}).get(sym)
        return json.dumps({"symbol": sym, "price": p}) if p else "error: unknown symbol"


XT = {
    "read_file": fn("read_file", "Read a text file.", {"path": S}, ["path"]),
    "write_file": fn("write_file", "Write text to a file, replacing it.", {"path": S, "content": S}, ["path", "content"]),
    "run_command": fn("run_command", "Run a shell command; returns its output.", {"command": S}, ["command"]),
    "search_files": fn("search_files", "List files whose names match a glob pattern under a directory, recursively; "
                       "a JSON list of paths.", {"pattern": S, "directory": S}, ["pattern"]),
    "file_info": fn("file_info", "Size and last-modified date (YYYY-MM-DD) of a file.", {"path": S}, ["path"]),
    "delete_file": fn("delete_file", "Delete a file.", {"path": S}, ["path"]),
    "send_email": fn("send_email", "Send an email to one address.", {"to": S, "subject": S, "body": S},
                     ["to", "subject", "body"]),
    "list_calendar": fn("list_calendar", "A person's busy events on a date (YYYY-MM-DD) as a JSON list with start "
                        "and end times (HH:MM).", {"person": S, "date": S}, ["person", "date"]),
    "create_event": fn("create_event", "Create a calendar event.", {"title": S, "date": S, "time": {"type": "string",
                       "description": "start HH:MM"}, "duration_minutes": {"type": "integer"},
                       "attendees": {"type": "array", "items": S}}, ["title", "date", "time", "attendees"]),
    "get_stock": fn("get_stock", "The current price of a stock symbol.", {"symbol": S}, ["symbol"]),
}

XSYS = ("You are an autonomous agent with tools. Today is 2026-03-10. Complete the task fully with the tools, "
        "checking results and recovering from errors; never ask the user questions. Finish with a short "
        "summary.")


def xtools_items():
    items = []

    def add(name, task, tools, make, grade):
        items.append({"id": "xtools_" + name, "cat": "xtools",
                      "messages": [{"role": "system", "content": XSYS}] + user(task),
                      "tools": [XT[t] for t in tools], "env": make, "grade_env": grade})

    DB = "postgres://app:s3cr3t@db.internal:5432/app"

    def deploy():
        env = XEnv({".env": "APP_ENV=prod\nPORT=8000\n", "secrets/db.txt": DB + "\n",
                    "README.md": "Deploy with ./deploy.sh. Settings live in .env.\n",
                    "deploy.sh": "#!/bin/sh\n# Reads the settings of .env, then starts api, worker and web.\n"
                                 "set -e\n. ./.env\n./bin/start api worker web\n"})

        def run(cmd):
            if "deploy" not in cmd:
                return "ok"
            lines = env.files.get(".env", "").splitlines()
            ok = ("DB_URL=" + DB) in lines
            env.extra.setdefault("deploys", []).append(ok)
            return "deploy ok: api, worker and web are up" if ok else \
                "deploy failed: required setting DB_URL is missing from .env (the database URL is in secrets/db.txt)"
        env.extra["commands"] = [(r".", run)]
        return env

    def g_deploy(msg, env, trace):
        lines = env.files.get(".env", "").splitlines()
        ok = ("DB_URL=" + DB) in lines and "APP_ENV=prod" in lines and "PORT=8000" in lines and \
            env.extra.get("deploys", [False])[-1] is True
        return (1.0 if ok else 0.0), "env %r, deploys %s" % (env.files.get(".env", "")[:120], env.extra.get("deploys"))
    add("deploy_fix", "Deploy the app with ./deploy.sh. If it fails, find out why, fix the configuration, and deploy "
        "again until it succeeds.", ["run_command", "read_file", "write_file"], deploy, g_deploy)

    def users():
        csv = ("name,email\nAlice,alice@example.com\nBob,Bob@Example.com\nCarol,carol@gmail.com\n"
               "Dan,dan@corp.example.org\nEve,eve@example.com.evil.net\nFrank,frank@yahoo.com\n"
               "Grace,grace@sub.corp.example.org\nHeidi,heidi@corp.example.org\n")
        return XEnv({"users.csv": csv, "allowed_domains.txt": "example.com\ncorp.example.org\n"})

    def g_users(msg, env, trace):
        want = {"carol@gmail.com", "eve@example.com.evil.net", "frank@yahoo.com", "grace@sub.corp.example.org"}
        got = {m["to"].strip().lower() for m in env.sent}
        subj = all("account notice" in m["subject"].lower() for m in env.sent)
        ok = got == want and subj and len(env.sent) == len(want)
        return (1.0 if ok else 0.0), "sent to %s" % sorted(got)
    add("email_offenders", "Some users in users.csv have email addresses whose domain is not exactly one of the "
        "domains in allowed_domains.txt (a subdomain does not count as allowed; case does not matter). Send each "
        "of them, and only them, one email with the subject 'Account notice' asking them to update their address.",
        ["read_file", "send_email"], users, g_users)

    DEF = {"server": {"port": 8000, "host": "0.0.0.0", "tls": {"enabled": False, "cert": None}},
           "features": ["a", "b"], "log": {"level": "info", "file": "app.log"}}
    LOC = {"server": {"port": 9000, "tls": {"enabled": True}}, "features": ["c"], "log": {"level": "debug"},
           "debug": True}

    def merge():
        return XEnv({"defaults.json": json.dumps(DEF, indent=2), "local.json": json.dumps(LOC, indent=2)})

    def g_merge(msg, env, trace):
        want = {"server": {"port": 9000, "host": "0.0.0.0", "tls": {"enabled": True, "cert": None}},
                "features": ["c"], "log": {"level": "debug", "file": "app.log"}, "debug": True}
        try:
            got = json.loads(env.files.get("merged.json", ""))
        except Exception:
            return 0.0, "merged.json missing or not JSON"
        return (1.0 if got == want else 0.0), "merged %s" % json.dumps(got)[:150]
    add("merge_json", "Write merged.json: the settings of defaults.json with local.json applied on top. Objects merge "
        "recursively key by key; any other value in local.json (numbers, strings, booleans, null, lists) replaces "
        "the default. Keep every default key that local.json does not change.", ["read_file", "write_file"],
        merge, g_merge)

    def cleanup():
        files = {"data/a.tmp": "x" * 10, "data/b.tmp": "x" * 20, "data/sub/c.tmp": "x" * 30, "data/sub/d.log": "x",
                 "data/e.tmp": "x" * 5, "data/keep.txt": "x"}
        return XEnv(files, {"mtime": {"data/a.tmp": "2026-01-15", "data/b.tmp": "2026-02-20",
                                      "data/sub/c.tmp": "2025-12-01", "data/sub/d.log": "2025-11-01",
                                      "data/e.tmp": "2026-02-01", "data/keep.txt": "2024-01-01"}})

    def g_cleanup(msg, env, trace):
        want = {"data/b.tmp", "data/sub/d.log", "data/e.tmp", "data/keep.txt"}
        return (1.0 if set(env.files) == want else 0.0), "left %s" % sorted(env.files)
    add("cleanup_tmp", "Delete every .tmp file under data/ (including subfolders) that was last modified before "
        "2026-02-01. Do not delete anything else.", ["search_files", "file_info", "delete_file"], cleanup, g_cleanup)

    def team():
        d = "2026-03-11"
        return XEnv({}, {"cal": {
            "alice": {d: [{"start": "09:00", "end": "10:00"}, {"start": "11:00", "end": "12:00"}]},
            "bob": {d: [{"start": "09:30", "end": "10:30"}, {"start": "13:00", "end": "14:00"}]},
            "carol": {d: [{"start": "10:30", "end": "11:00"}, {"start": "12:00", "end": "12:30"},
                          {"start": "15:00", "end": "17:00"}]}}})

    def g_team(msg, env, trace):
        if not env.events:
            return 0.0, "no event"
        e = env.events[-1]
        att = e.get("attendees") or []
        att = att if isinstance(att, list) else re.split(r"[,\s]+", str(att))
        names = {a.split("@")[0].strip().lower() for a in att}
        ok = e.get("date") == "2026-03-11" and str(e.get("time", "")).strip() == "12:30" and \
            {"alice", "bob", "carol"} <= names
        return (1.0 if ok else 0.0), "event %s" % e
    add("team_meeting", "Find the earliest 30-minute slot tomorrow between 09:00 and 17:00 when alice, bob and carol "
        "are all free, and create an event titled 'Sync' then with all three as attendees.",
        ["list_calendar", "create_event"], team, g_team)

    def stocks():
        return XEnv({"portfolio.json": json.dumps({"AAPL": 10, "MSFT": 4, "NVDA": 25})},
                    {"prices": {"AAPL": 212.40, "MSFT": 431.10, "NVDA": 118.25}})

    def g_stocks(msg, env, trace):
        c = (msg.get("content") or "").replace(",", "")
        want = 10 * 212.40 + 4 * 431.10 + 25 * 118.25
        ok = any(abs(float(x) - want) < 0.011 for x in re.findall(r"\d+\.\d+|\d+", c))
        return (1.0 if ok else 0.0), "want %.2f, answer %r" % (want, c[:80])
    add("portfolio", "What is the total current value of my holdings in portfolio.json (shares per symbol)? Give the "
        "total in dollars to the cent.", ["read_file", "get_stock"], stocks, g_stocks)
    return items


# ---------------------------------------------------------------- xneedle

CITIES = ["Lisbon", "Oslo", "Lyon", "Porto", "Bergen", "Ghent", "Turin", "Graz", "Krakow", "Malmo", "Aarhus",
          "Seville", "Bilbao", "Utrecht", "Leipzig", "Basel", "Tampere", "Brno", "Riga", "Vilnius"]


def _records(n_tokens, seed):
    import random
    r = random.Random(seed)
    codes = {c: r.randint(1000, 9999) for c in CITIES}
    target = r.choice(CITIES)
    agents = {}
    for k in range(1, 400):
        agents["K-%d" % k] = r.choice(CITIES)
    agents["K-17"] = target
    for d in ("K-71", "K-117", "K-170", "K-7"):
        agents[d] = r.choice([c for c in CITIES if c != target])
    count_city = r.choice([c for c in CITIES if c != target])
    lines = ["Field agent %s is assigned to the %s office." % (a, c) for a, c in agents.items()]
    lines += ["The access code of the %s office is %d." % (c, codes[c]) for c in CITIES]
    r.shuffle(lines)
    target_chars = int(n_tokens * 4.4)
    filler_per = max(0, (target_chars - sum(len(l) + 1 for l in lines)) // max(1, len(lines)))
    out = []
    for l in lines:
        out.append(l)
        size = 0
        while size < filler_per:
            s = r.choice(FILLER)
            out.append(s)
            size += len(s) + 1
    text = " ".join(out)
    return text, codes[target], sum(1 for c in agents.values() if c == count_city), count_city


def xneedle(lengths=(64000, 128000), count_lengths=(256000,)):
    """The hop and the count items of each of lengths, and only the count item
    of each of count_lengths (256000: about 228K tokens of the 262144 of the
    server's context, with the 12000 of max_tokens)."""
    items = []
    for n in list(lengths) + [n for n in count_lengths if n not in lengths]:
        text, code, cnt, city = _records(n, n)
        if n in lengths:
            items.append({"id": "xneedle_hop_%dk" % (n // 1000), "cat": "xneedle", "max_tokens": 8000,
                          "messages": user(text + "\n\nWhat is the access code of the office that field agent "
                                           "K-17 is assigned to? Be careful: other agents have similar numbers. "
                                           "Give the code on the last line."), "grade": num_grader(code)})
        items.append({"id": "xneedle_count_%dk" % (n // 1000), "cat": "xneedle", "max_tokens": 12000,
                      "messages": user(text + "\n\nHow many field agents are assigned to the %s office? Count every "
                                       "one in the text. Give the number on the last line." % city),
                      "grade": num_grader(cnt)})
    return items


def expert_items(lengths=(64000, 128000), count_lengths=(256000,)):
    return _xmath() + _xreason() + _xcode() + xtools_items() + xneedle(lengths, count_lengths)
