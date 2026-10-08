"""Check the graders and environments of hard.py with known answers: reference
solutions of the hcode items pass (and a wrong one fails); a scripted ideal
agent run of each htools task scores 1 and a run that does nothing scores 0;
good and bad answers of the hinstruct checks; the keys of hmath, hreason,
hneedle (printed for a look). Run after editing hard.py:

    python tests/quality/validate_hard.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hard  # noqa: E402

items = {it["id"]: it for it in hard.hard_items((8000,))}
bad = 0
cats = {}
for it in items.values():
    cats[it["cat"]] = cats.get(it["cat"], 0) + 1
print("items:", len(items), cats)

REF = {
    "eval_expr": '''
def eval_expr(s):
    toks = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace(): i += 1; continue
        if c.isdigit() or c == '.':
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'): j += 1
            toks.append(float(s[i:j])); i = j; continue
        toks.append(c); i += 1
    pos = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def take(): t = toks[pos[0]]; pos[0] += 1; return t
    def expr():
        v = term()
        while peek() in ('+', '-'):
            v = v + term() if take() == '+' else v - term()
        return v
    def term():
        v = unary()
        while peek() in ('*', '/'):
            v = v * unary() if take() == '*' else v / unary()
        return v
    def unary():
        if peek() == '-': take(); return -unary()
        if peek() == '+': take(); return unary()
        return atom()
    def atom():
        t = take()
        if t == '(':
            v = expr(); take(); return v
        return t
    return expr()
''',
    "edit_distance": '''
def edit_distance(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]
''',
    "lis_length": '''
import bisect
def lis_length(nums):
    t = []
    for x in nums:
        i = bisect.bisect_left(t, x)
        if i == len(t): t.append(x)
        else: t[i] = x
    return len(t)
''',
    "regex_match": '''
from functools import lru_cache
def regex_match(s, p):
    @lru_cache(None)
    def m(i, j):
        if j == len(p): return i == len(s)
        first = i < len(s) and p[j] in (s[i], '.')
        if j + 1 < len(p) and p[j + 1] == '*':
            return m(i, j + 2) or (first and m(i + 1, j))
        return first and m(i + 1, j + 1)
    return m(0, 0)
''',
    "topo_sort": '''
def topo_sort(n, edges):
    from collections import deque
    adj = [[] for _ in range(n)]; deg = [0] * n
    for u, v in edges: adj[u].append(v); deg[v] += 1
    q = deque(i for i in range(n) if deg[i] == 0); out = []
    while q:
        u = q.popleft(); out.append(u)
        for v in adj[u]:
            deg[v] -= 1
            if deg[v] == 0: q.append(v)
    return out if len(out) == n else None
''',
    "word_break": '''
def word_break(s, words):
    w = set(words); ok = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        ok[i] = any(ok[j] and s[j:i] in w for j in range(i))
    return ok[-1]
''',
    "median_finder": '''
import heapq
class MedianFinder:
    def __init__(self): self.lo = []; self.hi = []
    def add(self, x):
        heapq.heappush(self.lo, -x); heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo): heapq.heappush(self.lo, -heapq.heappop(self.hi))
    def median(self):
        return -self.lo[0] if len(self.lo) > len(self.hi) else (-self.lo[0] + self.hi[0]) / 2
''',
    "decode_ways": '''
def decode_ways(s):
    if not s: return 0
    a, b = 1, 0 if s[0] == '0' else 1
    for i in range(1, len(s)):
        c = b if s[i] != '0' else 0
        if s[i - 1] == '1' or (s[i - 1] == '2' and s[i] <= '6'): c += a
        a, b = b, c
    return b
''',
    "n_queens": '''
def n_queens(n):
    def go(r, cols, d1, d2):
        if r == n: return 1
        t = 0
        for c in range(n):
            if c not in cols and r - c not in d1 and r + c not in d2:
                t += go(r + 1, cols | {c}, d1 | {r - c}, d2 | {r + c})
        return t
    return go(0, set(), set(), set())
''',
    "knapsack": '''
def knapsack(weights, values, cap):
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1): dp[c] = max(dp[c], dp[c - w] + v)
    return dp[cap]
''',
    "spiral": '''
def spiral(m):
    out = []
    m = [list(r) for r in m]
    while m:
        out += m.pop(0)
        m = [list(r) for r in zip(*m)][::-1]
    return out
''',
    "min_window": '''
from collections import Counter
def min_window(s, t):
    if not t: return ''
    need = Counter(t); miss = len(t); best = (float('inf'), 0, 0); i = 0
    for j, c in enumerate(s, 1):
        if need[c] > 0: miss -= 1
        need[c] -= 1
        if miss == 0:
            while need[s[i]] < 0: need[s[i]] += 1; i += 1
            if j - i < best[0]: best = (j - i, i, j)
            need[s[i]] += 1; miss += 1; i += 1
    return s[best[1]:best[2]] if best[0] != float('inf') else ''
''',
    "meeting_rooms": '''
import heapq
def min_rooms(meetings):
    h = []
    for s, e in sorted(meetings):
        if h and h[0] <= s: heapq.heapreplace(h, e)
        else: heapq.heappush(h, e)
    return len(h)
''',
    "parse_duration": '''
import re
def parse_duration(s):
    m = re.fullmatch(r'(?:(\\d+)d)?(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', s)
    if not s or not m or not any(m.groups()): raise ValueError(s)
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se
''',
}
for name, code in REF.items():
    it = items["hcode_" + name]
    sc, note = it["grade"]({"content": "```python\n" + code + "\n```"})
    if sc < 1:
        print("HCODE FAIL", name, note); bad += 1
    if it["grade"]({"content": "```python\ndef nope(): pass\n```"})[0] > 0:
        print("HCODE accepts wrong", name); bad += 1

for iid, it in sorted(items.items()):
    if it["cat"] in ("hmath", "hreason", "hneedle") and "num" in repr(it["grade"]):
        pass
    if it["cat"] in ("hmath", "hneedle"):
        note = it["grade"]({"content": "0"})[1]
        print("  %-11s %s" % (iid, note.split(",")[0]))


def call(env, name, **a):
    return env.call(name, a)


IDEAL = {
    "largest_log": lambda e: ([call(e, "search_files", pattern="*.log", directory="/var/app")] +
                              [call(e, "file_info", path=p) for p in ["var/app/a.log", "var/app/old/b.log", "var/app/d.log"]],
                              "The largest is /var/app/old/b.log at 912,340 bytes."),
    "edit_config": lambda e: ([call(e, "read_file", path="config.json"),
                               call(e, "write_file", path="config.json", content=json.dumps(
                                   dict(json.loads(e.files["config.json"]), port=8080), indent=2))], "Done."),
    "office_weather": lambda e: ([call(e, "read_file", path="office.txt"), call(e, "get_weather", city="Lyon")],
                                 "It is 9°C in Lyon."),
    "count_todos": lambda e: ([call(e, "search_files", pattern="*.py", directory="src")] +
                              [call(e, "read_file", path=p) for p in json.loads(e.t_search_files("*.py", "src"))],
                              "There are 3 TODO comments."),
    "email_summary": lambda e: ([call(e, "read_file", path="report.txt"),
                                 call(e, "send_email", to="alice@example.com", subject="Report",
                                      body="Q1 revenue rose 12% to $4.2M while churn fell to 3.1%.")], "Sent."),
    "failing_test": lambda e: ([call(e, "run_command", command="pytest")], "test_migration_rollback failed."),
    "schedule": lambda e: ([call(e, "list_events", date="2026-03-11"),
                            call(e, "create_event", title="Planning", date="2026-03-11", time="11:00")], "Booked 11:00."),
    "currency": lambda e: ([call(e, "get_rate", base="USD", quote="EUR")], "250 USD is 229.35 EUR."),
    "host_down": lambda e: ([call(e, "read_file", path="hosts.txt")] +
                            [call(e, "run_command", command="ping -c 3 " + h) for h in
                             ["web1.internal", "web2.internal", "db1.internal", "db2.internal", "cache.internal"]],
                            "db2.internal is down."),
    "rename_jpeg": lambda e: ([call(e, "search_files", pattern="*.jpeg", directory="photos")] +
                              [call(e, "rename_file", src=p, dst=p[:-5] + ".jpg") for p in
                               json.loads(e.t_search_files("*.jpeg", "photos"))], "Renamed 3 files."),
}
for name, ideal in IDEAL.items():
    it = items["htools_" + name]
    env = it["env"]()
    _outs, final = ideal(env)
    sc, note = it["grade_env"]({"content": final}, env, [])
    if sc < 1:
        print("HTOOLS ideal FAIL", name, note); bad += 1
    env2 = it["env"]()
    sc2, _ = it["grade_env"]({"content": "I could not do it."}, env2, [])
    if sc2 > 0:
        print("HTOOLS accepts nothing", name); bad += 1

GOOD = {
    "lamp": "A slim desk lamp with a steel arm. Its warm glow suits late reading. The base holds a USB port.",
    "json_array": '[{"name":"a","score":90},{"name":"b","score":75},{"name":"c","score":75},{"name":"d","score":3}]',
    "acrostic": "Ash-grey skies hang low\nBare trees shiver\nCold winds wander\nDrifts pile deep\nEvening comes early",
    "french": "La tour Eiffel est célèbre pour son architecture en fer audacieuse. Elle domine Paris depuis 1889.",
    "sorted_list": "1. C\n2. Go\n3. Java\n4. Python\n5. Rust\n6. Swift",
    "seven_words": "The moon hangs low above the sea\nIts silver light falls on the waves\nQuiet tides answer "
                   "with a soft hush\nNight keeps its watch until the dawn",
    "table": "| Country | Capital |\n|---|---|\n| France | Paris |\n| Italy | Rome |\n| Japan | Tokyo |",
    "banana": "ananab\nBANANA\n6",
    "py_dict": "{'a': 1, 'b': ['x', 'y', 'z'], 'c': None}",
    "no_comma": "A vaccine teaches the immune system to spot a germ. It shows the body a harmless piece of the "
                "germ. The body then makes antibodies and memory cells. Later the real germ is fought off fast.",
}
for n, txt in GOOD.items():
    it = items["hinstruct_" + n]
    sc, note = it["grade"]({"content": txt})
    if sc < 1:
        print("HINSTRUCT FAIL", n, note); bad += 1
    if it["grade"]({"content": "nope, nope, nope, light virus " * 20})[0] > 0:
        print("HINSTRUCT accepts bad", n); bad += 1

for iid in [i for i in items if i.startswith("hreason")]:
    print("  %-11s %s" % (iid, items[iid]["grade"]({"content": "Z 0"})[1].split(",")[0]))
print("VALID" if bad == 0 else "PROBLEMS %d" % bad)
