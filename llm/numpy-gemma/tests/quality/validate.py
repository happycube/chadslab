"""Check the graders of suite.py with known answers: reference solutions of
the code items pass their tests and a wrong one fails, the math keys, the
letters, the instruction checks (a good and a bad answer), the tool graders,
the needle. Run after editing suite.py: python tests/quality/validate.py"""
import sys, json
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import suite
items = {it["id"]: it for it in suite.all_items((2000,))}
print("items:", len(items), {c: sum(1 for i in items.values() if i["cat"] == c) for c in ["math","code","tools","instruct","mcq","needle"]})
REF = {
"is_palindrome": "def is_palindrome(s):\n    t=[c.lower() for c in s if c.isalnum()]\n    return t==t[::-1]",
"fizzbuzz": "def fizzbuzz(n):\n    return ['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else str(i) for i in range(1,n+1)]",
"merge_intervals": "def merge_intervals(iv):\n    out=[]\n    for a,b in sorted(iv):\n        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n        else: out.append([a,b])\n    return out",
"roman_to_int": "def roman_to_int(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t=0\n    for i,c in enumerate(s):\n        if i+1<len(s) and v[c]<v[s[i+1]]: t-=v[c]\n        else: t+=v[c]\n    return t",
"word_freq": "import re\ndef word_freq(text):\n    d={}\n    for w in re.findall('[a-z]+', text.lower()): d[w]=d.get(w,0)+1\n    return d",
"flatten": "def flatten(x):\n    out=[]\n    for e in x:\n        if isinstance(e,(list,tuple)): out+=flatten(e)\n        else: out.append(e)\n    return out",
"binary_search": "def binary_search(a,x):\n    lo,hi=0,len(a)-1\n    while lo<=hi:\n        m=(lo+hi)//2\n        if a[m]==x: return m\n        if a[m]<x: lo=m+1\n        else: hi=m-1\n    return -1",
"lru_cache_class": "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self,c): self.c=c; self.d=OrderedDict()\n    def get(self,k):\n        if k not in self.d: return -1\n        self.d.move_to_end(k); return self.d[k]\n    def put(self,k,v):\n        self.d[k]=v; self.d.move_to_end(k)\n        if len(self.d)>self.c: self.d.popitem(last=False)",
"valid_parens": "def valid_parens(s):\n    st=[]; p={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in p:\n            if not st or st.pop()!=p[c]: return False\n    return not st",
"primes_upto": "def primes_upto(n):\n    return [i for i in range(2,n+1) if all(i%d for d in range(2,int(i**0.5)+1))]",
"rle": "from itertools import groupby\ndef rle(s):\n    return ''.join(f'{len(list(g))}{k}' for k,g in groupby(s))",
"matrix_mult": "def matmul(a,b):\n    return [[sum(x*y for x,y in zip(r,c)) for c in zip(*b)] for r in a]",
"topk_words": "from collections import Counter\ndef top_k(words,k):\n    c=Counter(words)\n    return sorted(c, key=lambda w:(-c[w],w))[:k]",
"dijkstra": "import heapq\ndef shortest(graph,src,dst):\n    d={src:0}; h=[(0,src)]\n    while h:\n        x,u=heapq.heappop(h)\n        if u==dst: return x\n        if x>d.get(u,1e300): continue\n        for v,w in graph.get(u,{}).items():\n            if x+w<d.get(v,float('inf')): d[v]=x+w; heapq.heappush(h,(x+w,v))\n    return float('inf')",
"to_camel": "import re\ndef to_camel(s):\n    p=re.split('[_-]',s)\n    return p[0]+''.join(w.capitalize() for w in p[1:])",
}
bad = 0
for name, code in REF.items():
    sc, note = items["code_" + name]["grade"]({"content": "```python\n" + code + "\n```"})
    if sc < 1: print("CODE FAIL", name, note); bad += 1
    sc, _ = items["code_" + name]["grade"]({"content": "```python\ndef nope(): pass\n```"})
    if sc > 0: print("CODE accepts wrong", name); bad += 1
import math
for iid, it in items.items():
    if it["cat"] == "math":
        want = float(it["grade"]({"content": "0"})[1].split(",")[0].split()[1])
        ok, _ = it["grade"]({"content": "Some work...\nThe answer is %s" % (want if want != int(want) else int(want))})
        if ok < 1: print("MATH FAIL", iid); bad += 1
        print("  %s = %s" % (iid, want)) if want != int(want) else None
MCQ_ANS = [a for _q, _o, a in suite.MCQ]
for i, a in enumerate(MCQ_ANS):
    it = items["mcq_%02d" % (i + 1)]
    if it["grade"]({"content": a})[0] < 1 or it["grade"]({"content": "Z"})[0] > 0: print("MCQ FAIL", i); bad += 1
good = {"three_bullets": "- a\n- b\n- c", "upper": "THE CAT SLEEPS ON THE MAT.", "json_only": '{"name": "Ana", "age": 30}',
        "word_limit": "A black hole is a region where gravity is so strong nothing escapes.", "start_with": "Golden leaves fall.",
        "no_letter_e": "Big surf rolls on a calm dark sand at night.", "end_with": "Avoid screens. Sweet dreams.",
        "numbered": "1. Boil\n2. Pour\n3. Steep\n4. Remove\n5. Drink", "lowercase": "soft rain falls\non quiet roofs\nnight listens",
        "german": "Der größte Planet ist Jupiter.", "two_paragraphs": "Trees grow.\n\nThey give shade.", "keyword_3": "The aroma, the aroma, oh the aroma.",
        "csv": "city,country\nParis,France\nRome,Italy\nOslo,Norway", "one_word": "Yellow", "quote": '"Picasso"'}
for n, txt in good.items():
    sc, note = items["instruct_" + n]["grade"]({"content": txt})
    if sc < 1: print("INSTRUCT FAIL", n, note); bad += 1
    sc, note = items["instruct_" + n]["grade"]({"content": ("NOPE " if n == "lowercase" else "nope ") * 40})
    if sc > 0: print("INSTRUCT accepts bad", n); bad += 1
def call(name, a): return {"tool_calls": [{"function": {"name": name, "arguments": json.dumps(a)}}], "content": ""}
T = {"weather": call("get_weather", {"city": "Lisbon", "unit": "celsius"}), "weather_f": call("get_weather", {"city": "Phoenix", "unit": "fahrenheit"}),
     "read": call("read_file", {"path": "/etc/hostname"}), "write": call("write_file", {"path": "notes/todo.txt", "content": "buy milk"}),
     "cmd": call("run_command", {"command": "df -h"}), "search": call("search_files", {"pattern": "*.py", "directory": "src"}),
     "event": call("create_event", {"title": "Dentist appointment", "date": "2026-03-14", "time": "15:30"}),
     "email": call("send_email", {"to": "bob@example.com", "subject": "Lunch", "body": "See you at noon."}),
     "none_math": {"content": "12 times 12 is 144."}, "none_fact": {"content": "Paris."}, "none_write": {"content": "Waves.\nSalt."},
     "use_weather": {"content": "It is -7°C with light snow in Oslo."}, "use_file": {"content": "Version 4.12.7"},
     "use_cmd": {"content": "data.csv has 83,521 lines."}, "use_search": {"content": "Yes: /var/app/a.log and /var/app/old/b.log."}}
for n, m in T.items():
    sc, note = items["tools_" + n]["grade"](m)
    if sc < 1: print("TOOLS FAIL", n, note); bad += 1
nd = [i for i in items.values() if i["cat"] == "needle"][0]
import re
phrase = re.search(r"passphrase for the archive is (\S+)\.", nd["messages"][0]["content"]).group(1)
if nd["grade"]({"content": phrase})[0] < 1: print("NEEDLE FAIL"); bad += 1
print("needle chars for 2000 tokens:", len(nd["messages"][0]["content"]))
print("VALID" if bad == 0 else "PROBLEMS %d" % bad)
