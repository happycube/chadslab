"""The think part and the end of the context in an answer, token by token
(Sampler.guard): ThinkGuard closes a think part at a budget (and when the
model ends the turn inside it), and puts in the words to wrap up when few
tokens of the context are left. Every engine uses it: Qwen, Nemotron and
DeepSeek (<think> ... </think>), Gemma 4 (<|channel>thought ... <channel|>).
The engines take each token of the answer through Sampler.fixed, the drafts
of speculative rounds too: a token the guard puts in a draft's place ends the
drafts taken there.
"""
from __future__ import annotations

import os
import sys


def think_left_open(prompt_ids, special):
    """True when the last turn of the prompt opens <think> and does not close
    it: the answer starts in the think part."""
    op, cl, st = special.get("<think>"), special.get("</think>"), special.get("<|im_start|>")
    if op is None or cl is None:
        return False
    ids = list(prompt_ids)
    start = max((i for i, t in enumerate(ids) if t == st), default=-1)
    tail = ids[start + 1:]
    return op in tail and cl not in tail[tail.index(op):]


class ThinkGuard:
    """The tokens of an answer, fixed as they come (Sampler.guard;
    Sampler.fixed: once for each token, in order).

    The think part of a prompt that ends in <think>: while it is open, an
    end of the turn (<|im_end|>, <|endoftext|>) becomes </think>, so the
    model goes on to write the answer. A long think ended so: 16748 tokens
    of reasoning (47532 chars, a coding agent's prompt of 8824 tokens)
    ending "Let's write.<|im_end|>", and the client got no answer.

    budget (--think-budget): after that many tokens of the think part, the
    tokens of force (THINK_CLOSE) take the place of the picked ones, one a
    token, up to </think>; the model then writes the answer. With ends_line
    (a test of the tokens whose text ends in a newline: newline_test), they
    wait for the end of a line, as NVIDIA's budget for Nemotron: the first
    token from the budget on that ends a line goes in, then force without
    its leading newlines; no line ended in NL_WAIT tokens (at most a quarter
    of the budget), force where it is. A close at the budget alone cut
    "So (8" in the middle. para (DeepSeek): they wait for an empty line in
    place of the end of a line (a token from the budget on after which the
    text ends in two newlines, spaces between them too): the end of a
    paragraph of the think part. A coding
    agent's client closed six requests at 303 s each (about 9600 tokens of
    the model at 32 tok/s), and a think of 15339 tokens took 632 s.

    wrap_at (--wrap-left): at that token of the answer (fewer than
    --wrap-left tokens of the context left), the tokens of wrap_think (in
    the think part) or wrap_answer take the place of the picked ones, once;
    not inside a tool call (tool: the ids of <tool_call>, </tool_call>),
    whose text they would break: then at its end. A coding agent's session
    went on to 89033 tokens of a context of 98304.

    close_now: a loop in the think part closes it as the budget does (once);
    the repeat test of server.py had ended the turn there, with no answer
    (the Gemma 26B: "*Salt liquid flows in vast ways.*" again and again).

    open_id: the token that opens a think part in the answer (Gemma 4: the
    model writes <|channel>thought ... <channel|> itself); think is then the
    state at the start (a prompt that ends in an open channel)."""

    def __init__(self, stop_ids, close_id, budget=0, force=(), think=True, wrap_at=None,
                 wrap_think=(), wrap_answer=(), tool=(None, None), open_id=None, ends_line=None,
                 para=False):
        self.stop, self.close, self.open, self.fixes = set(stop_ids), close_id, bool(think), 0
        self.open_id = open_id
        self.budget, self.force, self.n, self.queue, self.budget_hits = budget, list(force), 0, [], 0
        self.wrap_at, self.wrap_think, self.wrap_answer = wrap_at, list(wrap_think), list(wrap_answer)
        self.tool_open, self.tool_close = tool
        self.count, self.in_tool, self.wrapped = 0, False, False
        self.loops = 0
        self.ends_line = ends_line
        self.wait = min(NL_WAIT, budget // 4) if ends_line is not None else 0
        self.waited = 0
        # para: the newlines that end the text of the think part (an empty line: 2)
        self.para, self.nls = bool(para) and ends_line is not None, 0

    def __call__(self, token):
        self.count += 1
        t = self._fix(token)
        if t == self.close:
            self.open = False
        elif self.open_id is not None and t == self.open_id:
            self.open = True
        if self.tool_open is not None and t == self.tool_open:
            self.in_tool = True
        elif self.tool_close is not None and t == self.tool_close:
            self.in_tool = False
        return t

    def close_now(self):
        """A loop in the think part (the repeat test of server.py): the
        words of the budget (force) take the place of the next tokens, up to
        the close, once. Return True when they do (or still go in); False
        when the think part is not open, or a loop closed it before."""
        if self.queue:
            return True
        if not self.open or not self.force or self.loops:
            return False
        self.loops += 1
        self.queue = list(self.force)
        return True

    def _fix(self, token):
        if self.queue:
            return self.queue.pop(0)
        if (self.wrap_at is not None and not self.wrapped and self.count >= self.wrap_at
                and not self.in_tool):
            words = self.wrap_think if self.open else self.wrap_answer
            if words:
                self.wrapped = True
                self.queue = list(words)
                return self.queue.pop(0)
        if self.open and token in self.stop:
            self.open = False
            self.fixes += 1
            return self.close
        if token == self.close or not self.open:
            return token
        self.n += 1
        if self.para and self.n < self.budget:
            # (the newlines that end the text before the budget: the first token on may end an empty line)
            self.nls = newlines_after(self.ends_line, token, self.nls)
        if self.budget and self.n >= self.budget and self.force:
            self.waited = self.n - self.budget
            if self.waited < self.wait:
                if self.para:
                    self.nls = newlines_after(self.ends_line, token, self.nls)
                if not self.ends_line(token) or self.para and self.nls < 2:
                    return token
                # (the line ends: the token, then force from its first word)
                self.budget_hits += 1
                k = 0
                while k < len(self.force) - 1 and self.ends_line(self.force[k]):
                    k += 1
                self.queue = self.force[k:]
                return token
            self.budget_hits += 1
            self.queue = list(self.force)
            return self.queue.pop(0)
        return token

    @staticmethod
    def of(prompt_ids, special, stop_ids, budget=0, force=(), wrap_at=None, wrap_think=(),
           wrap_answer=(), ends_line=None):
        """A guard when the last turn of the prompt opens <think> (and does
        not close it), or with wrap_at; else None."""
        cl = special.get("</think>")
        think = think_left_open(prompt_ids, special)
        if not think and wrap_at is None:
            return None
        return ThinkGuard(stop_ids, cl, budget, force, think=think, wrap_at=wrap_at,
                          wrap_think=wrap_think if cl is not None else (), wrap_answer=wrap_answer,
                          tool=(special.get("<tool_call>"), special.get("</tool_call>")),
                          ends_line=ends_line)


# The least think part that a request of enough max_tokens gets (the half of
# max_tokens alone gave 3000 to a request of 6000: too little for the hard
# items), RECLANK_THINK_MIN; ANSWER_ROOM tokens are left for the answer.
THINK_MIN = int(os.environ.get("RECLANK_THINK_MIN", "6000"))
ANSWER_ROOM = 1000
# the most tokens past the budget that a think part goes on to the end of a line (NVIDIA's 500 for
# Nemotron's budget), RECLANK_THINK_NL_WAIT; 0: the close at the budget
NL_WAIT = int(os.environ.get("RECLANK_THINK_NL_WAIT", "500"))


def think_budget(budget, max_tokens, n_force):
    """The budget of a think part: --think-budget (0: none) at most, and of
    max_tokens (never raised) the larger of half and THINK_MIN, with
    ANSWER_ROOM tokens and the THINK_CLOSE tokens (n_force) left for the
    answer. A client's title request (max_tokens 64) thought for all 64
    tokens and got an empty answer; now it thinks for 15 or so, then
    THINK_CLOSE, then 32 tokens of answer. A request of 6000: 4980."""
    half = max(1, max_tokens // 2 - n_force)
    cap = max(half, min(THINK_MIN, max_tokens - n_force - ANSWER_ROOM))
    return min(budget, cap) if budget > 0 else cap


def newline_test(decode):
    """ThinkGuard's ends_line from a tokenizer's decode: True for a token whose text ends in a
    newline (each token decoded once, when the guard first asks). ends_line.tail(token): the
    newlines of the spaces that end its text, and whether its text is all spaces (newlines_after)."""
    seen = {}

    def text(token):
        r = seen.get(token)
        if r is None:
            r = seen[token] = decode([int(token)])
        return r

    def ends_line(token):
        return text(token).endswith("\n")

    def tail(token):
        t = text(token)
        rest = t.rstrip()
        return t[len(rest):].count("\n"), not rest
    ends_line.tail = tail
    return ends_line


def newlines_after(ends_line, token, nls):
    """The newlines that end the text after token (ThinkGuard para), nls before it: a token of
    spaces alone adds its own; another, those that end it."""
    k, blank = ends_line.tail(token)
    return nls + k if blank else k


def wrap_at_of(wrap_left, room, max_tokens):
    """The token of the answer from which fewer than wrap_left (--wrap-left)
    positions of the context are left, room: the positions after the prompt;
    None when max_tokens does not get there."""
    if wrap_left > 0 and max_tokens > room - wrap_left:
        return max(1, room - wrap_left + 1)
    return None


def report(guard, tag, wrap_left):
    """The lines of the log for what a guard did in an answer (none: None)."""
    if guard is None:
        return
    if guard.fixes:
        print("[%s] the model ended the turn inside the think part: its close in its place" % tag,
              file=sys.stderr, flush=True)
    if guard.wrapped:
        print("[%s] fewer than %d tokens of the context left at token %d of the answer: the words to "
              "wrap up" % (tag, wrap_left, guard.wrap_at), file=sys.stderr, flush=True)
    if guard.budget_hits:
        at = (" at the end of a %s, %d tokens on" % ("paragraph" if guard.para else "line", guard.waited)
              if guard.waited < guard.wait else
              " (no %s ended in %d tokens)" % ("paragraph" if guard.para else "line", guard.wait)
              if guard.wait else "")
        print("[%s] the think part reached its budget (%d tokens): closed%s" % (tag, guard.budget, at),
              file=sys.stderr, flush=True)
