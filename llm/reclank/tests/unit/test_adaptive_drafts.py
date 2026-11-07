"""speculative.AdaptiveDrafts: each count measured COST_MIN times first (the start first), then the
count with the most tokens a second from the acceptance of the recent rounds and the round times
(Bolt E4B's on the 3090), and a switch when the acceptance changes."""
from reclank.core.speculative import AdaptiveDrafts

# the measured time of a round of k drafts (Bolt E4B, DFlash, the 3090)
COST = {1: 0.027, 2: 0.029, 3: 0.032, 4: 0.040, 5: 0.044, 6: 0.048, 7: 0.053}


def run(ad, rounds, kept_of):
    picks = []
    for _ in range(rounds):
        k = ad.pick()
        picks.append(k)
        ad.update(k, min(k, kept_of(k)), COST[k])
    return picks


def test_explores_each_count_from_the_start():
    ad = AdaptiveDrafts(range(1, 8), start=3)
    picks = run(ad, 21, lambda k: k)
    assert picks[:3] == [3, 3, 3]
    assert sorted(set(picks)) == list(range(1, 8))
    assert all(len(ad.times[k]) == AdaptiveDrafts.COST_MIN for k in ad.ks)


def test_long_drafts_kept_takes_the_best_rate():
    ad = AdaptiveDrafts(range(1, 8), start=3)
    run(ad, 21, lambda k: 0)
    # every draft kept: k + 1 tokens a round; the best (k + 1) / COST[k] is 7 (8 / 0.053)
    assert run(ad, 30, lambda k: k)[-1] == max(COST, key=lambda k: (k + 1) / COST[k])


def test_few_kept_takes_few_drafts_and_switches_back():
    ad = AdaptiveDrafts(range(1, 8), start=3)
    run(ad, 21, lambda k: k)
    # one draft kept a round (a short chat): 1 or 2 drafts
    assert run(ad, 40, lambda k: 1)[-1] in (1, 2)
    # then code-like rounds: more drafts again
    assert run(ad, 40, lambda k: k)[-1] >= 5


def test_unreached_drafts_take_the_last_acceptance():
    ad = AdaptiveDrafts(range(1, 8), start=2)
    for _ in range(20):
        ad.update(2, 1, COST[2])
    a = ad.accept()
    assert abs(a[1] - 1.0) < 1e-9 and a[2] < 0.01
    assert a[3:] == [a[2]] * 5
