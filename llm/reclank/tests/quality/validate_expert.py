"""Check expert.py: independent solutions of the xcode items pass their hidden
tests (and a wrong one fails); a scripted ideal agent run of each xtools task
scores 1 and an empty one 0; the keys of xmath, xreason, xneedle printed.

    python tests/quality/validate_expert.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert  # noqa: E402

items = {it["id"]: it for it in expert.expert_items((8000,))}
bad = 0

SOL = {
    "json_parse": r'''
def json_parse(s):
    i = 0
    n = len(s)
    def ws():
        nonlocal i
        while i < n and s[i] in ' \t\n\r': i += 1
    def val():
        nonlocal i
        ws()
        if i >= n: raise ValueError('end')
        c = s[i]
        if c == '{':
            i += 1; ws(); d = {}
            if i < n and s[i] == '}': i += 1; return d
            while True:
                ws()
                if i >= n or s[i] != '"': raise ValueError('key')
                k = string(); ws()
                if i >= n or s[i] != ':': raise ValueError(':')
                i += 1; d[k] = val(); ws()
                if i < n and s[i] == ',': i += 1; continue
                if i < n and s[i] == '}': i += 1; return d
                raise ValueError('obj')
        if c == '[':
            i += 1; ws(); a = []
            if i < n and s[i] == ']': i += 1; return a
            while True:
                a.append(val()); ws()
                if i < n and s[i] == ',': i += 1; continue
                if i < n and s[i] == ']': i += 1; return a
                raise ValueError('arr')
        if c == '"': return string()
        for lit, v in (('true', True), ('false', False), ('null', None)):
            if s.startswith(lit, i): i += len(lit); return v
        return number()
    def string():
        nonlocal i
        i += 1; out = []
        esc = {'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}
        while True:
            if i >= n: raise ValueError('unterminated')
            c = s[i]
            if c == '"': i += 1; return ''.join(out)
            if c == '\\':
                i += 1
                if i >= n: raise ValueError('esc')
                e = s[i]
                if e in esc: out.append(esc[e]); i += 1
                elif e == 'u':
                    h = s[i + 1:i + 5]
                    if len(h) != 4 or any(x not in '0123456789abcdefABCDEF' for x in h): raise ValueError('u')
                    out.append(chr(int(h, 16))); i += 5
                else: raise ValueError('bad escape')
            elif ord(c) < 32: raise ValueError('control')
            else: out.append(c); i += 1
    def number():
        nonlocal i
        st = i
        if i < n and s[i] == '-': i += 1
        if i < n and s[i] == '0': i += 1
        elif i < n and s[i] in '123456789':
            while i < n and s[i].isdigit(): i += 1
        else: raise ValueError('num')
        fl = False
        if i < n and s[i] == '.':
            i += 1; fl = True
            if not (i < n and s[i].isdigit()): raise ValueError('frac')
            while i < n and s[i].isdigit(): i += 1
        if i < n and s[i] in 'eE':
            i += 1; fl = True
            if i < n and s[i] in '+-': i += 1
            if not (i < n and s[i].isdigit()): raise ValueError('exp')
            while i < n and s[i].isdigit(): i += 1
        t = s[st:i]
        return float(t) if fl else int(t)
    v = val(); ws()
    if i != n: raise ValueError('trailing')
    return v
''',
    "sudoku": '''
def solve(grid):
    cells = [[0 if ch == '.' else int(ch) for ch in r] for r in grid]
    rows = [set() for _ in range(9)]; cols = [set() for _ in range(9)]; boxes = [set() for _ in range(9)]
    empty = []
    for i in range(9):
        for j in range(9):
            v = cells[i][j]
            if v: rows[i].add(v); cols[j].add(v); boxes[i//3*3+j//3].add(v)
            else: empty.append((i, j))
    def go():
        best, bc = None, None
        for (i, j) in empty:
            if cells[i][j]: continue
            c = [v for v in range(1, 10) if v not in rows[i] and v not in cols[j] and v not in boxes[i//3*3+j//3]]
            if best is None or len(c) < len(bc):
                best, bc = (i, j), c
                if len(c) <= 1: break
        if best is None: return True
        i, j = best
        for v in bc:
            cells[i][j] = v; rows[i].add(v); cols[j].add(v); boxes[i//3*3+j//3].add(v)
            if go(): return True
            cells[i][j] = 0; rows[i].discard(v); cols[j].discard(v); boxes[i//3*3+j//3].discard(v)
        return False
    go()
    return [''.join(map(str, r)) for r in cells]
''',
    "lisp": '''
def eval_lisp(src):
    toks = src.replace('(', ' ( ').replace(')', ' ) ').split()
    pos = 0
    def read():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t == '(':
            out = []
            while toks[pos] != ')': out.append(read())
            pos += 1
            return out
        try: return int(t)
        except ValueError: return t
    import operator, functools
    G = {'+': lambda *a: sum(a), '*': lambda *a: functools.reduce(operator.mul, a, 1),
         '-': lambda a, b=None: -a if b is None else a - b, '<': operator.lt, '>': operator.gt, '=': operator.eq}
    def ev(x, env):
        if isinstance(x, int): return x
        if isinstance(x, str):
            if x == '#t': return True
            if x == '#f': return False
            e = env
            while e is not None:
                if x in e[0]: return e[0][x]
                e = e[1]
            raise NameError(x)
        h = x[0]
        if h == 'define':
            if isinstance(x[1], list):
                name, params = x[1][0], x[1][1:]
                env[0][name] = ('fn', params, x[2], env)
            else:
                env[0][x[1]] = ev(x[2], env)
            return None
        if h == 'lambda': return ('fn', x[1], x[2], env)
        if h == 'if': return ev(x[2], env) if ev(x[1], env) is not False else ev(x[3], env)
        f = ev(h, env); args = [ev(a, env) for a in x[1:]]
        if isinstance(f, tuple) and f[0] == 'fn':
            return ev(f[2], (dict(zip(f[1], args)), f[3]))
        return f(*args)
    env = (dict(G), None)
    v = None
    while pos < len(toks): v = ev(read(), env)
    return v
''',
    "skyline": '''
import heapq
def skyline(b):
    ev = sorted([(l, -h, r) for l, r, h in b] + [(r, 0, 0) for _, r, _ in b])
    res, live = [], [(0, float('inf'))]
    for x, nh, r in ev:
        while live[0][1] <= x: heapq.heappop(live)
        if nh: heapq.heappush(live, (nh, r))
        h = -live[0][0]
        if not res or res[-1][1] != h:
            if res and res[-1][0] == x: res[-1][1] = h
            else: res.append([x, h])
            if len(res) > 1 and res[-1][1] == res[-2][1]: res.pop()
    return res
''',
    "pal_subseq": '''
def count_pal_subseq(s):
    M = 10**9 + 7
    from functools import lru_cache
    import sys
    sys.setrecursionlimit(10000)
    n = len(s)
    nxt = [[n]*4 for _ in range(n+1)]; prv = [[-1]*4 for _ in range(n+1)]
    for i in range(n-1, -1, -1):
        nxt[i] = nxt[i+1][:]; nxt[i][ord(s[i])-97] = i
    for i in range(n):
        prv[i+1] = prv[i][:]; prv[i+1][ord(s[i])-97] = i
    dp = [[0]*(n+1) for _ in range(n+1)]
    for length in range(1, n+1):
        for i in range(0, n-length+1):
            j = i + length
            t = 0
            for c in range(4):
                a, b = nxt[i][c], prv[j][c]
                if a >= j: continue
                if a == b: t += 1
                else: t += 2 + dp[a+1][b]
            dp[i][j] = t % M
    return dp[0][n]
''',
    "grid_k": '''
from collections import deque
def shortest_k(grid, k):
    m, n = len(grid), len(grid[0])
    best = {(0, 0): k}
    q = deque([(0, 0, k, 0)])
    while q:
        i, j, r, d = q.popleft()
        if i == m-1 and j == n-1: return d
        for a, b in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
            if 0 <= a < m and 0 <= b < n:
                rr = r - grid[a][b]
                if rr >= 0 and best.get((a, b), -1) < rr:
                    best[(a, b)] = rr; q.append((a, b, rr, d+1))
    return -1
''',
    "trap_2d": '''
import heapq
def trap_2d(h):
    if not h or not h[0]: return 0
    m, n = len(h), len(h[0]); vis = set(); pq = []
    for i in range(m):
        for j in range(n):
            if i in (0, m-1) or j in (0, n-1): heapq.heappush(pq, (h[i][j], i, j)); vis.add((i, j))
    tot = 0
    while pq:
        v, i, j = heapq.heappop(pq)
        for a, b in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
            if 0 <= a < m and 0 <= b < n and (a, b) not in vis:
                vis.add((a, b)); tot += max(0, v - h[a][b]); heapq.heappush(pq, (max(v, h[a][b]), a, b))
    return tot
''',
    "alien_order": '''
def alien_order(words):
    from collections import deque
    letters = []
    for w in words:
        for c in w:
            if c not in letters: letters.append(c)
    g = {c: set() for c in letters}; deg = {c: 0 for c in letters}
    for a, b in zip(words, words[1:]):
        for x, y in zip(a, b):
            if x != y:
                if y not in g[x]: g[x].add(y); deg[y] += 1
                break
        else:
            if len(a) > len(b): return ''
    q = deque(c for c in letters if deg[c] == 0); out = []
    while q:
        c = q.popleft(); out.append(c)
        for d in g[c]:
            deg[d] -= 1
            if deg[d] == 0: q.append(d)
    return ''.join(out) if len(out) == len(letters) else ''
''',
    "median_two": '''
def median_two(a, b):
    if len(a) > len(b): a, b = b, a
    m, n = len(a), len(b); lo, hi = 0, m; half = (m + n + 1) // 2
    while lo <= hi:
        i = (lo + hi) // 2; j = half - i
        al = a[i-1] if i > 0 else float('-inf'); ar = a[i] if i < m else float('inf')
        bl = b[j-1] if j > 0 else float('-inf'); br = b[j] if j < n else float('inf')
        if al <= br and bl <= ar:
            if (m + n) % 2: return float(max(al, bl))
            return (max(al, bl) + min(ar, br)) / 2
        if al > br: hi = i - 1
        else: lo = i + 1
''',
    "regex_engine": '''
def full_match(s, p):
    pos = 0
    def parse_alt():
        nonlocal pos
        node = parse_seq()
        while pos < len(p) and p[pos] == '|':
            pos += 1; node = ('alt', node, parse_seq())
        return node
    def parse_seq():
        nonlocal pos
        items = []
        while pos < len(p) and p[pos] not in '|)':
            items.append(parse_post())
        return ('seq', items)
    def parse_post():
        nonlocal pos
        if p[pos] == '(':
            pos += 1; a = parse_alt(); pos += 1
        else:
            a = ('ch', p[pos]); pos += 1
        while pos < len(p) and p[pos] in '*+?':
            a = (p[pos], a); pos += 1
        return a
    tree = parse_alt()
    from functools import lru_cache
    def m(node, i):
        t = node[0]
        if t == 'ch':
            return {i + 1} if i < len(s) and (node[1] == '.' or s[i] == node[1]) else set()
        if t == 'seq':
            cur = {i}
            for it in node[1]:
                cur = set().union(*[m(it, j) for j in cur]) if cur else set()
            return cur
        if t == 'alt': return m(node[1], i) | m(node[2], i)
        if t == '?': return {i} | m(node[1], i)
        if t in '*+':
            out = set() if t == '+' else {i}
            frontier = m(node[1], i)
            while frontier - out:
                new = frontier - out; out |= new
                frontier = set().union(*[m(node[1], j) for j in new])
            return out
    return len(s) in m(tree, 0)
''',
}
for name, code in SOL.items():
    it = items["xcode_" + name]
    sc, note = it["grade"]({"content": "```python\n" + code + "\n```"})
    if sc < 1:
        print("XCODE FAIL", name, note); bad += 1
    if it["grade"]({"content": "```python\ndef nope(): pass\n```"})[0] > 0:
        print("XCODE accepts wrong", name); bad += 1


def call(env, name, **a):
    return env.call(name, a)


def ideal_deploy(e):
    call(e, "run_command", command="./deploy.sh")
    db = call(e, "read_file", path="secrets/db.txt").strip()
    call(e, "write_file", path=".env", content=e.files[".env"] + "DB_URL=" + db + "\n")
    call(e, "run_command", command="./deploy.sh")
    return "Deployed after adding DB_URL."


def ideal_users(e):
    for to in ["carol@gmail.com", "eve@example.com.evil.net", "frank@yahoo.com", "grace@sub.corp.example.org"]:
        call(e, "send_email", to=to, subject="Account notice", body="Please update your address.")
    return "Sent 4."


def ideal_merge(e):
    d, l = json.loads(e.files["defaults.json"]), json.loads(e.files["local.json"])

    def mg(a, b):
        out = dict(a)
        for k, v in b.items():
            out[k] = mg(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
        return out
    call(e, "write_file", path="merged.json", content=json.dumps(mg(d, l)))
    return "Wrote merged.json."


def ideal_cleanup(e):
    for p in json.loads(call(e, "search_files", pattern="*.tmp", directory="data")):
        if json.loads(call(e, "file_info", path=p))["modified"] < "2026-02-01":
            call(e, "delete_file", path=p)
    return "Deleted 2."


def ideal_team(e):
    for who in ("alice", "bob", "carol"):
        call(e, "list_calendar", person=who, date="2026-03-11")
    call(e, "create_event", title="Sync", date="2026-03-11", time="12:30", attendees=["alice", "bob", "carol"])
    return "Booked 12:30."


def ideal_stocks(e):
    pf = json.loads(call(e, "read_file", path="portfolio.json"))
    tot = 0
    for s, n in pf.items():
        r = call(e, "get_stock", symbol=s)
        if r.startswith("error"):
            r = call(e, "get_stock", symbol=s)
        tot += n * json.loads(r)["price"]
    return "Total: $%.2f" % tot


IDEAL = {"deploy_fix": ideal_deploy, "email_offenders": ideal_users, "merge_json": ideal_merge,
         "cleanup_tmp": ideal_cleanup, "team_meeting": ideal_team, "portfolio": ideal_stocks}
for name, f in IDEAL.items():
    it = items["xtools_" + name]
    env = it["env"]()
    final = f(env)
    sc, note = it["grade_env"]({"content": final}, env, [])
    if sc < 1:
        print("XTOOLS ideal FAIL", name, note); bad += 1
    env2 = it["env"]()
    if it["grade_env"]({"content": "I could not."}, env2, [])[0] > 0:
        print("XTOOLS accepts nothing", name); bad += 1

for iid, it in sorted(items.items()):
    if it["cat"] in ("xmath", "xreason", "xneedle"):
        print("  %-18s %s" % (iid, it["grade"]({"content": "zzz 0"})[1].split(",")[0]))
print("VALID" if bad == 0 else "PROBLEMS %d" % bad)
