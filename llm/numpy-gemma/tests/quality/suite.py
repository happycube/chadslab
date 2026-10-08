"""The items of the quality suite (scripts/quality_suite.py) and their graders.

Each item is a dict: id, cat (the category), messages (the chat), and a
grader grade(msg) -> (score 0..1, note), where msg is the message of the
answer ({"content", "reasoning_content", "tool_calls"}). Some items carry
tools (OpenAI function schemas) or max_tokens. The answers that are numbers
are computed here, not typed, so the key has no slips.

Categories:
    math         word problems with one exact number as the answer
    code         a Python function, graded by unit tests run in a subprocess
    tools        the right tool and arguments, no tool when none fits, and an
                 answer from a tool result
    instruct     instructions with a checkable form (count, case, JSON, ...)
    mcq          multiple choice: knowledge and logic, one letter
    needle       a fact in a long text (built by the runner at several lengths)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------- helpers


def user(text):
    return [{"role": "user", "content": text}]


def last_number(text):
    """The last number of the text (commas in thousands removed), or None."""
    t = (text or "").replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    return float(nums[-1]) if nums else None


def plain_number(text):
    """The number that is all of text (markup aside: **, $, `, \\boxed{},
    \\, , commas, a final period), or None."""
    t = re.sub(r"\\boxed\{|\\text\{|\\[,!; ]|[*$`{},\s]", "", text or "").rstrip(".")
    return float(t) if re.fullmatch(r"-?\d+(?:\.\d+)?", t) else None


def boxed_or_last(text):
    """The number of the last line when it is only a number (the prompts ask
    for that), else that of the last \\boxed{...} that holds only a number
    (not a formula: \\boxed{a(n) = 4a(n-2) - a(n-4)} gave -4), else the one
    after 'answer', else the last number."""
    t = text or ""
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if lines:
        v = plain_number(lines[-1])
        if v is not None:
            return v
    m = [v for v in (plain_number(b) for b in re.findall(r"\\boxed\{([^}]*)\}", t)) if v is not None]
    if m:
        return m[-1]
    m = re.findall(r"(?i)answer\s*[:=is]*\s*\$?(-?[\d,]+(?:\.\d+)?)", t)
    if m:
        return float(m[-1].replace(",", ""))
    return last_number(t)


def num_grader(want, tol=1e-6):
    def g(msg):
        got = boxed_or_last(msg.get("content"))
        ok = got is not None and abs(got - want) <= tol * max(1.0, abs(want))
        return (1.0 if ok else 0.0), "want %s, got %s" % (want, got)
    return g


MATH_TAIL = " Give the final answer as a single number on the last line."


# ---------------------------------------------------------------- math

def _math():
    items = []

    def add(q, ans):
        items.append({"id": "math_%02d" % (len(items) + 1), "cat": "math",
                      "messages": user(q + MATH_TAIL), "grade": num_grader(ans)})
    add("A shop sells pencils at 3 for $1.20. How many dollars do 25 pencils cost?", 25 * 1.20 / 3)
    add("A train travels 342 km in 3 hours and then 228 km in 2 hours. What is its average "
        "speed in km/h over the whole trip?", (342 + 228) / 5)
    add("What is 17 * 23 + 144 / 12?", 17 * 23 + 144 / 12)
    add("A rectangle has a perimeter of 54 cm and its length is twice its width. What is its "
        "area in square cm?", 18 * 9)
    add("Tom has 4 times as many marbles as Sara. Together they have 85 marbles. How many "
        "marbles does Tom have?", 68)
    add("What is the sum of all integers from 1 to 250?", sum(range(1, 251)))
    add("A price of $80 is raised by 25% and then the new price is lowered by 20%. What is "
        "the final price in dollars?", 80 * 1.25 * 0.8)
    add("How many seconds are there in 3 days, 4 hours and 5 minutes?",
        3 * 86400 + 4 * 3600 + 5 * 60)
    add("A tank holds 1200 liters. It is filled by a pipe at 45 liters per minute while a "
        "leak drains 15 liters per minute. Starting empty, how many minutes until it is "
        "full?", 1200 / 30)
    add("What is the greatest common divisor of 462 and 1071?", 21)
    add("A recipe needs 2.5 cups of flour for 12 cookies. How many cups are needed for 54 "
        "cookies?", 2.5 * 54 / 12)
    add("The average of five numbers is 18. Four of them are 12, 20, 25 and 9. What is the "
        "fifth number?", 18 * 5 - (12 + 20 + 25 + 9))
    add("A car uses 6.5 liters of fuel per 100 km. Fuel costs $1.80 per liter. What does a "
        "trip of 420 km cost in dollars?", 4.2 * 6.5 * 1.80)
    add("How many ways can you choose 3 people from a group of 10?", 120)
    add("What is 2 to the power of 20?", 2 ** 20)
    add("A clock shows 3:15. What is the smaller angle in degrees between the hour and "
        "minute hands?", 7.5)
    add("Alice reads 18 pages a day. Her book has 412 pages and she has read 106. How many "
        "more full days does she need to finish it (count a partial day as a day)?",
        -(-(412 - 106) // 18))
    add("A worker earns $22 per hour for the first 40 hours of a week and 1.5 times that "
        "for each hour beyond 40. How much does she earn in a 47-hour week, in dollars?",
        40 * 22 + 7 * 33)
    add("What is the least common multiple of 18, 24 and 30?", 360)
    add("A store has 240 apples. It sells 25% on Monday and 40% of the rest on Tuesday. How "
        "many apples are left?", 240 * 0.75 * 0.6)
    add("Solve for x: 7x - 12 = 3x + 20.", 8)
    add("A box has 6 red, 4 blue and 10 green balls. What percent of the balls are not "
        "green?", 50)
    add("If a sequence starts 3, 8, 13, 18, ..., what is its 40th term?", 3 + 39 * 5)
    add("A square garden of side 14 m has a path 1 m wide around the outside of it. What is "
        "the area of the path in square meters?", 16 * 16 - 14 * 14)
    add("Three friends split a bill of $157.50 so that the first pays twice as much as each "
        "of the other two, who pay equal amounts. How much does the first friend pay?",
        157.5 / 2)
    return items


# ---------------------------------------------------------------- code

CODE_TAIL = (" Reply with the complete function in a single ```python code block; no tests, "
             "no example usage.")

CODE = [
    ("is_palindrome", "Write a Python function is_palindrome(s: str) -> bool that returns True "
     "when s reads the same forwards and backwards, ignoring case and any character that is not "
     "a letter or digit.",
     ["assert is_palindrome('A man, a plan, a canal: Panama')",
      "assert not is_palindrome('race a car')", "assert is_palindrome('')",
      "assert is_palindrome('No lemon, no melon!')", "assert not is_palindrome('ab')"]),
    ("fizzbuzz", "Write a Python function fizzbuzz(n: int) -> list[str] that returns the strings "
     "for 1..n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for multiples of "
     "both, else the number as a string.",
     ["assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']",
      "assert fizzbuzz(15)[-1] == 'FizzBuzz'", "assert len(fizzbuzz(100)) == 100",
      "assert fizzbuzz(0) == []"]),
    ("merge_intervals", "Write a Python function merge_intervals(iv: list[list[int]]) -> "
     "list[list[int]] that merges overlapping closed intervals and returns them sorted by start. "
     "Intervals that touch (like [1,2] and [2,3]) merge.",
     ["assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
      "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]", "assert merge_intervals([]) == []",
      "assert merge_intervals([[5,7],[1,2],[2,4]]) == [[1,4],[5,7]]"]),
    ("roman_to_int", "Write a Python function roman_to_int(s: str) -> int that converts a Roman "
     "numeral (I, V, X, L, C, D, M with subtractive forms) to an integer.",
     ["assert roman_to_int('III') == 3", "assert roman_to_int('LVIII') == 58",
      "assert roman_to_int('MCMXCIV') == 1994", "assert roman_to_int('XLIX') == 49"]),
    ("word_freq", "Write a Python function word_freq(text: str) -> dict[str, int] that counts "
     "words case-insensitively, where a word is a maximal run of letters a-z (after lowercasing) "
     "and everything else separates words.",
     ["assert word_freq('Hello, hello world!') == {'hello': 2, 'world': 1}",
      "assert word_freq('') == {}", "assert word_freq(\"It's 3 o'clock\") == {'it': 1, 's': 1, 'o': 1, 'clock': 1}"]),
    ("flatten", "Write a Python function flatten(x) that flattens arbitrarily nested lists and "
     "tuples into a flat list, keeping the order; strings are not split.",
     ["assert flatten([1,[2,[3,(4,5)]],'ab']) == [1,2,3,4,5,'ab']", "assert flatten([]) == []",
      "assert flatten([[[]]]) == []"]),
    ("binary_search", "Write a Python function binary_search(a: list[int], x: int) -> int that "
     "returns the index of x in the sorted list a, or -1 if x is not present. Use binary search.",
     ["assert binary_search([1,3,5,7,9], 7) == 3", "assert binary_search([1,3,5,7,9], 4) == -1",
      "assert binary_search([], 1) == -1", "assert binary_search([2], 2) == 0",
      "assert binary_search(list(range(0, 1000, 2)), 998) == 499"]),
    ("lru_cache_class", "Write a Python class LRUCache with __init__(self, capacity: int), "
     "get(self, key) -> value or -1, and put(self, key, value). When full, put evicts the least "
     "recently used key; get and put both count as use.",
     ["c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1; c.put(3, 3); "
      "assert c.get(2) == -1; c.put(4, 4); assert c.get(1) == -1; assert c.get(3) == 3; "
      "assert c.get(4) == 4"]),
    ("valid_parens", "Write a Python function valid_parens(s: str) -> bool that checks that the "
     "brackets (), [] and {} in s are balanced and properly nested; other characters are "
     "ignored.",
     ["assert valid_parens('([]{})')", "assert not valid_parens('([)]')",
      "assert valid_parens('a(b)c')", "assert not valid_parens('((')", "assert valid_parens('')"]),
    ("primes_upto", "Write a Python function primes_upto(n: int) -> list[int] returning all primes "
     "<= n in increasing order.",
     ["assert primes_upto(10) == [2,3,5,7]", "assert primes_upto(1) == []",
      "assert len(primes_upto(1000)) == 168", "assert primes_upto(2) == [2]"]),
    ("rle", "Write a Python function rle(s: str) -> str that run-length encodes s as count "
     "followed by character for each run, e.g. 'aaabcc' -> '3a1b2c'.",
     ["assert rle('aaabcc') == '3a1b2c'", "assert rle('') == ''", "assert rle('z') == '1z'",
      "assert rle('aaaaaaaaaaaa') == '12a'"]),
    ("matrix_mult", "Write a Python function matmul(a: list[list[float]], b: list[list[float]]) "
     "-> list[list[float]] multiplying two matrices given as lists of rows, without numpy.",
     ["assert matmul([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]",
      "assert matmul([[1,2,3]], [[1],[2],[3]]) == [[14]]"]),
    ("topk_words", "Write a Python function top_k(words: list[str], k: int) -> list[str] that "
     "returns the k most frequent words, ordered by decreasing frequency and then "
     "alphabetically for ties.",
     ["assert top_k(['i','love','code','i','love','coding'], 2) == ['i','love']",
      "assert top_k(['b','a','c','a','b','c'], 2) == ['a','b']", "assert top_k([], 3) == []"]),
    ("dijkstra", "Write a Python function shortest(graph: dict, src, dst) -> float returning the "
     "length of the shortest path from src to dst, where graph maps a node to a dict of "
     "neighbor -> nonnegative edge weight; return float('inf') if dst is unreachable.",
     ["g = {'a': {'b': 1, 'c': 4}, 'b': {'c': 2, 'd': 5}, 'c': {'d': 1}, 'd': {}}",
      "assert shortest(g, 'a', 'd') == 4", "assert shortest(g, 'd', 'a') == float('inf')",
      "assert shortest(g, 'a', 'a') == 0"]),
    ("to_camel", "Write a Python function to_camel(s: str) -> str converting snake_case or "
     "kebab-case (or a mix) to lowerCamelCase, e.g. 'hello_big-world' -> 'helloBigWorld'.",
     ["assert to_camel('hello_big-world') == 'helloBigWorld'", "assert to_camel('x') == 'x'",
      "assert to_camel('make_it_work') == 'makeItWork'"]),
]


def extract_code(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return max(m, key=len) if m else (text or "")


def run_tests(code, tests, timeout=15):
    """Run the code and its asserts in a fresh python with limits. Return
    (passed, note)."""
    prog = code + "\n\n" + "\n".join(tests) + "\nprint('ALL_OK')\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        with open(path, "w") as f:
            f.write(prog)

        def limits():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True,
                               timeout=timeout + 5, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if "ALL_OK" in r.stdout:
        return True, "pass"
    err = (r.stderr or "").strip().splitlines()
    return False, err[-1][:160] if err else "no output"


def _code():
    items = []
    for name, q, tests in CODE:
        def g(msg, tests=tests):
            ok, note = run_tests(extract_code(msg.get("content")), tests)
            return (1.0 if ok else 0.0), note
        items.append({"id": "code_" + name, "cat": "code", "messages": user(q + CODE_TAIL),
                      "grade": g})
    return items


# ---------------------------------------------------------------- tools

def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}


S = {"type": "string"}
TOOLS = [
    fn("get_weather", "Current weather for a city.", {"city": S, "unit": {"type": "string",
       "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("read_file", "Read a text file and return its contents.", {"path": S}, ["path"]),
    fn("write_file", "Write text to a file, replacing it.", {"path": S, "content": S},
       ["path", "content"]),
    fn("run_command", "Run a shell command and return its output.", {"command": S}, ["command"]),
    fn("search_files", "Find files whose names match a glob pattern under a directory.",
       {"pattern": S, "directory": S}, ["pattern"]),
    fn("create_event", "Create a calendar event.", {"title": S, "date": {"type": "string",
       "description": "YYYY-MM-DD"}, "time": {"type": "string", "description": "HH:MM, 24h"}},
       ["title", "date"]),
    fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
]


def norm(v):
    return re.sub(r"\s+", " ", str(v).strip().lower())


def tool_grader(name, args=None, contains=None):
    """The first tool call: its name, and each argument of args equal (norm)
    or, for contains, holding the text."""
    args, contains = args or {}, contains or {}

    def g(msg):
        calls = msg.get("tool_calls") or []
        if not calls:
            return 0.0, "no tool call (content %r)" % (msg.get("content") or "")[:80]
        f = calls[0].get("function", {})
        if f.get("name") != name:
            return 0.0, "called %s" % f.get("name")
        try:
            a = json.loads(f.get("arguments") or "{}")
        except Exception:
            return 0.0, "arguments not JSON: %r" % (f.get("arguments") or "")[:80]
        bad = [k for k, v in args.items() if norm(a.get(k, "")) != norm(v)]
        bad += [k for k, v in contains.items() if norm(v) not in norm(a.get(k, ""))]
        return (0.0 if bad else 1.0), ("bad %s: %s" % (bad, a) if bad else "ok %s" % a)
    return g


def no_tool_grader(must=None):
    def g(msg):
        if msg.get("tool_calls"):
            return 0.0, "called %s" % msg["tool_calls"][0].get("function", {}).get("name")
        c = msg.get("content") or ""
        if must and not re.search(must, c, re.I):
            return 0.0, "answer without %r: %r" % (must, c[:80])
        return 1.0, "ok"
    return g


def _tools():
    sysm = {"role": "system", "content": "You are a helpful assistant with tools. Today is "
            "2026-03-10. Use a tool when it is needed; otherwise answer directly."}
    T = [
        ("weather", "What's the weather like in Lisbon right now, in celsius?",
         tool_grader("get_weather", {"city": "Lisbon", "unit": "celsius"})),
        ("weather_f", "Is it hot in Phoenix today? Give me Fahrenheit.",
         tool_grader("get_weather", {"city": "Phoenix", "unit": "fahrenheit"})),
        ("read", "Show me what's in /etc/hostname.", tool_grader("read_file", {"path": "/etc/hostname"})),
        ("write", "Create a file notes/todo.txt containing exactly the text: buy milk",
         tool_grader("write_file", {"path": "notes/todo.txt"}, {"content": "buy milk"})),
        ("cmd", "How much free disk space is there? Use a shell command.",
         tool_grader("run_command", contains={"command": "df"})),
        ("search", "Find all the Python files under src/.",
         tool_grader("search_files", contains={"pattern": ".py"})),
        ("event", "Put a dentist appointment on my calendar for March 14 at 3:30 pm.",
         tool_grader("create_event", {"date": "2026-03-14", "time": "15:30"}, {"title": "dentist"})),
        ("email", "Email bob@example.com with the subject 'Lunch' and say: see you at noon.",
         tool_grader("send_email", {"to": "bob@example.com", "subject": "Lunch"}, {"body": "noon"})),
        ("none_math", "What is 12 times 12?", no_tool_grader(r"\b144\b")),
        ("none_fact", "What is the capital of France?", no_tool_grader(r"paris")),
        ("none_write", "Write me a two-line poem about the sea.", no_tool_grader()),
    ]
    items = []
    for name, q, g in T:
        items.append({"id": "tools_" + name, "cat": "tools", "messages": [sysm] + user(q),
                      "tools": TOOLS, "grade": g})
    # an answer from a tool result
    res = [
        ("use_weather", "What's the weather in Oslo?", "get_weather", {"city": "Oslo"},
         {"city": "Oslo", "temp_c": -7, "conditions": "light snow"}, r"-7|minus 7|−7"),
        ("use_file", "What version is in VERSION?", "read_file", {"path": "VERSION"},
         "4.12.7\n", r"4\.12\.7"),
        ("use_cmd", "How many lines does data.csv have? Use wc.", "run_command",
         {"command": "wc -l data.csv"}, "  83521 data.csv\n", r"83,?521"),
        ("use_search", "Are there any .log files in /var/app?", "search_files",
         {"pattern": "*.log", "directory": "/var/app"}, ["/var/app/a.log", "/var/app/old/b.log"],
         r"a\.log|b\.log|two|2"),
    ]
    for name, q, tool, a, out, must in res:
        call = {"id": "call_1", "type": "function", "function": {"name": tool,
                "arguments": json.dumps(a)}}
        msgs = [sysm] + user(q) + [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(out) if not
             isinstance(out, str) else out}]
        items.append({"id": "tools_" + name, "cat": "tools", "messages": msgs, "tools": TOOLS,
                      "grade": no_tool_grader(must)})
    return items


# ---------------------------------------------------------------- instruct

def words(t):
    return re.findall(r"[A-Za-z0-9'’-]+", t or "")


def check(fn_, note):
    def g(msg):
        try:
            ok = bool(fn_(msg.get("content") or ""))
        except Exception as e:
            ok, note2 = False, repr(e)[:80]
            return 0.0, note2
        return (1.0 if ok else 0.0), note
    return g


def _json_keys(t, keys):
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    d = json.loads(t)
    return isinstance(d, dict) and set(d) == set(keys)


def _instruct():
    I = [
        ("three_bullets", "List exactly three benefits of exercise as a markdown bullet list, and "
         "nothing else.", lambda t: len(re.findall(r"(?m)^\s*[-*•] ", t)) == 3, "3 bullets"),
        ("upper", "Describe a cat in one sentence written entirely in capital letters.",
         lambda t: t.strip() and t.upper() == t and len(words(t)) >= 4, "all caps"),
        ("json_only", "Return only a JSON object with the keys \"name\" and \"age\" for a fictional "
         "person. No other text.", lambda t: _json_keys(t, ["name", "age"]), "JSON keys"),
        ("word_limit", "Explain what a black hole is in at most 30 words.",
         lambda t: 5 <= len(words(t)) <= 30, "<=30 words"),
        ("start_with", "Write one sentence about autumn that starts with the word 'Golden'.",
         lambda t: t.strip().lstrip("*_\"'").startswith("Golden"), "starts Golden"),
        ("no_letter_e", "Write a sentence of at least six words about the ocean without using the "
         "letter 'e'.", lambda t: len(words(t)) >= 6 and "e" not in t.lower(), "no e"),
        ("end_with", "Give a short tip for sleeping better. End your reply with the exact phrase "
         "'Sweet dreams.'", lambda t: t.strip().rstrip("*_\"'").endswith("Sweet dreams."), "ends"),
        ("numbered", "Give exactly 5 numbered steps to make tea (1. 2. ...), nothing else.",
         lambda t: re.findall(r"(?m)^\s*(\d+)[.)] ", t) == ["1", "2", "3", "4", "5"], "5 steps"),
        ("lowercase", "write a haiku about rain using only lowercase letters.",
         lambda t: t.strip() and t.lower() == t, "lowercase"),
        ("german", "Answer in German only: what is the largest planet in the solar system?",
         lambda t: re.search(r"(?i)jupiter", t) and re.search(r"(?i)\b(der|ist|planet|größte|groesste)\b", t)
         and not re.search(r"(?i)\b(the|largest|is)\b", t), "German"),
        ("two_paragraphs", "Write exactly two paragraphs about trees, separated by one blank line.",
         lambda t: len([p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]) == 2, "2 paras"),
        ("keyword_3", "Write a short paragraph about coffee that uses the word 'aroma' exactly "
         "three times.", lambda t: len(re.findall(r"(?i)\baroma\b", t)) == 3, "aroma x3"),
        ("csv", "Output a CSV with a header row 'city,country' and exactly three data rows. Only "
         "the CSV.", lambda t: (lambda L: L[0].replace(" ", "").lower() == "city,country" and len(L) == 4
                                and all(l.count(",") == 1 for l in L))(
             [l for l in re.sub(r"^```\w*|```$", "", t.strip()).strip().splitlines() if l.strip()]),
         "CSV 3 rows"),
        ("one_word", "Answer with a single word: what color is a ripe banana?",
         lambda t: len(words(t)) == 1 and "yellow" in t.lower(), "one word"),
        ("quote", "Wrap your entire answer in double quotation marks: name a famous painter.",
         lambda t: t.strip().startswith('"') and t.strip().endswith('"'), "quoted"),
    ]
    return [{"id": "instruct_" + n, "cat": "instruct", "messages": user(q), "grade": check(f, note)}
            for n, q, f, note in I]


# ---------------------------------------------------------------- mcq

MCQ = [
    ("What is the chemical symbol for gold?", ["Ag", "Au", "Gd", "Go"], "B"),
    ("Which planet has the shortest year?", ["Venus", "Mars", "Mercury", "Earth"], "C"),
    ("Who wrote 'Pride and Prejudice'?", ["Charlotte Brontë", "Jane Austen", "Mary Shelley",
                                          "George Eliot"], "B"),
    ("What is the derivative of x^3?", ["3x^2", "x^2", "3x", "x^4/4"], "A"),
    ("Which gas makes up most of Earth's atmosphere?", ["Oxygen", "Carbon dioxide", "Argon",
                                                         "Nitrogen"], "D"),
    ("In which year did World War II end?", ["1943", "1944", "1945", "1946"], "C"),
    ("What is the time complexity of binary search on a sorted array?", ["O(n)", "O(log n)",
                                                                          "O(n log n)", "O(1)"], "B"),
    ("Which organelle produces most of a cell's ATP?", ["Nucleus", "Ribosome", "Mitochondrion",
                                                         "Golgi apparatus"], "C"),
    ("All bloops are razzies and all razzies are lazzies. Which must be true?",
     ["All lazzies are bloops", "All bloops are lazzies", "No razzies are bloops",
      "Some lazzies are not razzies"], "B"),
    ("What is the boiling point of water at sea level in Fahrenheit?", ["100", "180", "212", "273"],
     "C"),
    ("Which data structure gives first-in, first-out order?", ["Stack", "Queue", "Heap",
                                                                 "Tree"], "B"),
    ("Which country has the largest population as of 2024?", ["China", "India", "USA",
                                                              "Indonesia"], "B"),
    ("What does HTTP status 404 mean?", ["Server error", "Unauthorized", "Not found",
                                         "Redirect"], "C"),
    ("If it is 3 pm in London (UTC+0), what time is it in Tokyo (UTC+9)?", ["6 am", "midnight",
                                                                             "noon", "6 pm"], "B"),
    ("Which is a prime number?", ["51", "57", "91", "97"], "D"),
    ("What is the SI unit of electric resistance?", ["Volt", "Ohm", "Ampere", "Watt"], "B"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does "
     "the ball cost?", ["$0.10", "$0.05", "$0.15", "$0.01"], "B"),
    ("Which language is primarily used for styling web pages?", ["HTML", "Python", "CSS", "SQL"],
     "C"),
    ("What is the largest ocean on Earth?", ["Atlantic", "Indian", "Arctic", "Pacific"], "D"),
    ("Which of these sorts is stable?", ["Quicksort (typical)", "Heapsort", "Merge sort",
                                         "Selection sort"], "C"),
]


def letter_grader(want):
    def g(msg):
        c = (msg.get("content") or "").strip()
        m = re.findall(r"\b([ABCD])\b", c)
        got = m[-1] if m else None
        return (1.0 if got == want else 0.0), "want %s, got %s" % (want, got)
    return g


def _mcq():
    items = []
    for i, (q, opts, ans) in enumerate(MCQ):
        text = q + "\n" + "\n".join("%s. %s" % ("ABCD"[j], o) for j, o in enumerate(opts)) + \
            "\nAnswer with only the letter of the correct option."
        items.append({"id": "mcq_%02d" % (i + 1), "cat": "mcq", "messages": user(text),
                      "grade": letter_grader(ans)})
    return items


# ---------------------------------------------------------------- needle

FILLER = [
    "The committee reviewed the quarterly figures and noted a modest rise in costs.",
    "Rain fell steadily over the valley while the river rose against its banks.",
    "A new library opened downtown, with a reading room on the top floor.",
    "Engineers replaced the old pumps, and the station ran quietly through the night.",
    "The museum added a gallery of maps drawn by sailors in the eighteenth century.",
    "Farmers in the region planted earlier this year because of the mild spring.",
    "The orchestra rehearsed the second movement until the tempo felt right.",
    "Traffic on the bridge slowed during the evening as fog drifted in from the bay.",
    "Students gathered in the courtyard to hear the results of the science fair.",
    "The bakery on the corner sold out of rye bread before nine in the morning.",
]


def needle_item(n_tokens, depth, seed):
    """A text of about n_tokens tokens (about 4.4 characters a token for this
    filler) with a passphrase at depth (0..1); the answer is the passphrase."""
    import random
    r = random.Random(seed)
    words_ = ["amber", "falcon", "seventeen", "copper", "lantern", "orchid", "granite", "violet",
              "harbor", "meadow", "cobalt", "ember"]
    phrase = "-".join(r.sample(words_, 3)) + "-%d" % r.randint(100, 999)
    target = int(n_tokens * 4.4)
    out, size = [], 0
    while size < target:
        s = r.choice(FILLER)
        out.append(s)
        size += len(s) + 1
    k = int(len(out) * depth)
    out.insert(k, "Remember this: the secret passphrase for the archive is %s." % phrase)
    text = " ".join(out)
    q = text + "\n\nWhat is the secret passphrase for the archive? Reply with the passphrase only."

    def g(msg):
        c = msg.get("content") or ""
        return (1.0 if phrase in c else 0.0), "want %s, got %r" % (phrase, c[:60])
    return {"id": "needle_%dk_d%02d" % (n_tokens // 1000, int(depth * 100)), "cat": "needle",
            "messages": user(q), "grade": g, "max_tokens": 1500}


def needle(lengths=(4000, 16000, 64000)):
    items = []
    for n in lengths:
        for d in (0.1, 0.5, 0.9):
            items.append(needle_item(n, d, seed=n + int(d * 100)))
    return items


def all_items(needle_lengths=(4000, 16000, 64000)):
    return _math() + _code() + _tools() + _instruct() + _mcq() + needle(needle_lengths)
