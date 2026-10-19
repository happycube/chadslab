/* ---------- HotCache (backends/cuda/runtime.py): the scores and the copies ----------
 * rc_hot_score is HotCache.score_group without its _replace, rc_hot_replace
 * the choice of the copies of HotCache._replace: the same float32 sums, the
 * same orders (the stable sorts of NumPy), so the same slots. In Python the
 * two took about 1.9 ms of each round of MTP decode of Qwen3.8-Flash-Next
 * (48 rows of 512 experts), while the GPU waited. The tables: m rows of E
 * experts, row-major; bool as uint8. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* score_group: the selections sels (n tokens x R rows x k; -1 for none) of
 * the rows rows. score *= decay^n, then decay^(n-1-j) for each selection of
 * token j in turn; uses, total, run get the counts; cand (R x E) gets the
 * experts that may take a slot. Return the count of cand. */
int rc_hot_score(float *score, int32_t *uses, int64_t *total, float *run, const uint8_t *held,
                 const uint8_t *incoming, const int64_t *rows, const int32_t *sels, int n, int R,
                 int E, int k, double decay, int run_mode, int pool_free, const uint8_t *pool_rows,
                 int admit, uint8_t *cand)
{
    const float dn = (float)pow(decay, (double)n);
    int32_t *cnt = (int32_t *)calloc((size_t)E, sizeof(int32_t));
    int nc = 0;
    for (int a = 0; a < R; ++a) {
        int64_t r = rows[a];
        float *sr = score + r * E;
        for (int x = 0; x < E; ++x) {
            sr[x] *= dn;
        }
        memset(cnt, 0, (size_t)E * sizeof(int32_t));
        for (int j = 0; j < n; ++j) {
            const float w = (float)pow(decay, (double)(n - 1 - j));
            const int32_t *s = sels + ((size_t)j * R + a) * k;
            for (int q = 0; q < k; ++q) {
                if (s[q] >= 0) {
                    sr[s[q]] += w;
                    ++cnt[s[q]];
                }
            }
        }
        int32_t *ur = uses + r * E;
        int64_t *tr = total + r * E;
        float *rr = run + r * E;
        const uint8_t *hr = held + r * E;
        for (int x = 0; x < E; ++x) {
            if (cnt[x]) {
                ur[x] += cnt[x];
                tr[x] += cnt[x];
                rr[x] += (float)cnt[x];
            }
            if (hr[x]) {
                ur[x] = 0;
            }
        }
        const float *key = run_mode ? rr : sr;
        float low = INFINITY;
        if (pool_free && pool_rows[r]) {
            low = -INFINITY;
        } else {
            for (int x = 0; x < E; ++x) {
                if (hr[x] && key[x] < low) {
                    low = key[x];
                }
            }
        }
        const uint8_t *ir = incoming + r * E;
        uint8_t *cr = cand + (size_t)a * E;
        for (int x = 0; x < E; ++x) {
            cr[x] = cnt[x] > 0 && !hr[x] && !ir[x] && key[x] > low && ur[x] >= admit;
            nc += cr[x];
        }
    }
    free(cnt);
    return nc;
}

/* The indices 0..n-1 in the order of v, ties by index (np.argsort
 * kind="stable"; -0 as +0, no NaN): an LSD radix sort of the bits of v as
 * unsigned keys in their order (a merge sort took 0.54 ms for the 5000
 * warm experts of the pool, most of a call). */
static void hot_argsort(const float *v, int n, int *idx)
{
    size_t nb = (size_t)(n > 0 ? n : 1);
    uint32_t *k = (uint32_t *)malloc(nb * sizeof(uint32_t)), *k2 = (uint32_t *)malloc(nb * sizeof(uint32_t));
    int *i2 = (int *)malloc(nb * sizeof(int));
    for (int i = 0; i < n; ++i) {
        uint32_t u = 0;
        if (v[i] != 0.0f) {
            memcpy(&u, v + i, sizeof(u));
        }
        k[i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        idx[i] = i;
    }
    for (int sh = 0; sh < 32; sh += 8) {
        int off[256] = {0};
        for (int i = 0; i < n; ++i) {
            ++off[(k[i] >> sh) & 255];
        }
        for (int d = 0, a = 0; d < 256; ++d) {
            int c = off[d];
            off[d] = a;
            a += c;
        }
        for (int i = 0; i < n; ++i) {
            int p = off[(k[i] >> sh) & 255]++;
            k2[p] = k[i];
            i2[p] = idx[i];
        }
        uint32_t *tk = k;
        k = k2;
        k2 = tk;
        memcpy(idx, i2, (size_t)n * sizeof(int));
    }
    free(k);
    free(k2);
    free(i2);
}

/* _replace: move up to limit experts of cand (R x E, the rows rows, any
 * order) to the slots of the lowest held experts (key: the score or the
 * uses of the conversation), best gain first; with a pool (owner, nblocks;
 * NULL for none) a row of pool_rows takes a free block, or the block of the
 * warm expert of the lowest key of any row. slots[r] is the slot table of
 * row r. The tables, slots, and owner change as in Python; items (limit x 3)
 * gets (row, expert, slot) of each copy, changed (m) the rows whose old
 * experts went cold. Return the count of items. */
int rc_hot_replace(const float *key, uint8_t *held, uint8_t *incoming, int32_t *const *slots,
                   int64_t *owner, int64_t nblocks, const uint8_t *pool_rows, const uint8_t *cand,
                   const int64_t *rows, int R, int m, int E, int limit, int64_t warm,
                   int64_t *items, uint8_t *changed)
{
    /* the candidates in the order of np.nonzero of the whole table */
    int *ord_r = (int *)malloc((size_t)R * sizeof(int));
    for (int a = 0; a < R; ++a) {
        ord_r[a] = a;
    }
    for (int a = 1; a < R; ++a) {          /* (a few rows: insertion by row) */
        int v = ord_r[a], b = a - 1;
        while (b >= 0 && rows[ord_r[b]] > rows[v]) {
            ord_r[b + 1] = ord_r[b];
            --b;
        }
        ord_r[b + 1] = v;
    }
    int nc = 0;
    for (int a = 0; a < R; ++a) {
        const uint8_t *c = cand + (size_t)ord_r[a] * E;
        for (int x = 0; x < E; ++x) {
            nc += c[x] != 0;
        }
    }
    int n_items = 0;
    if (nc == 0) {
        free(ord_r);
        return 0;
    }
    int *ri = (int *)malloc((size_t)nc * sizeof(int)), *xi = (int *)malloc((size_t)nc * sizeof(int));
    int *inv = (int *)malloc((size_t)nc * sizeof(int)), *ur = (int *)malloc((size_t)R * sizeof(int));
    int nu = 0, o = 0;
    for (int a = 0; a < R; ++a) {
        int r = (int)rows[ord_r[a]];
        const uint8_t *c = cand + (size_t)ord_r[a] * E;
        int first = 1;
        for (int x = 0; x < E; ++x) {
            if (c[x]) {
                if (first) {
                    ur[nu++] = r;
                    first = 0;
                }
                ri[o] = r;
                xi[o] = x;
                inv[o] = nu - 1;
                ++o;
            }
        }
    }
    /* the lowest held key of each row of the candidates (inf for none), its
     * expert (np.argmin: the first), and the gain order */
    float *hlow = (float *)malloc((size_t)nu * sizeof(float));
    int *va = (int *)malloc((size_t)nu * sizeof(int));
    int *urow = (int *)malloc((size_t)m * sizeof(int));
    for (int r = 0; r < m; ++r) {
        urow[r] = -1;
    }
    for (int u = 0; u < nu; ++u) {
        const float *kr = key + (size_t)ur[u] * E;
        const uint8_t *hr = held + (size_t)ur[u] * E;
        float lo = INFINITY;
        int at = 0;
        for (int x = 0; x < E; ++x) {
            float v = hr[x] ? kr[x] : INFINITY;
            if (v < lo) {
                lo = v;
                at = x;
            }
        }
        hlow[u] = lo;
        va[u] = at;
        urow[ur[u]] = u;
    }
    float *gain = (float *)malloc((size_t)nc * sizeof(float));
    for (int q = 0; q < nc; ++q) {
        gain[q] = hlow[inv[q]] - key[(size_t)ri[q] * E + xi[q]];
    }
    int *order = (int *)malloc((size_t)nc * sizeof(int));
    hot_argsort(gain, nc, order);
    /* the free blocks of the pool (the first limit), and the warm experts by
     * key once they are needed (pool_scan) */
    int use_pool = 0;
    if (owner != NULL) {
        for (int q = 0; q < nc && !use_pool; ++q) {
            use_pool = pool_rows[ri[q]] != 0;
        }
    }
    int64_t *freeb = (int64_t *)malloc((size_t)(limit > 0 ? limit : 1) * sizeof(int64_t));
    int nfree = 0, fpos = 0;
    if (use_pool) {
        for (int64_t b = 0; b < nblocks && nfree < limit; ++b) {
            if (owner[b] == -1) {
                freeb[nfree++] = b;
            }
        }
    }
    int scanned = !use_pool, have_pv = 0, npb = 0, pk = 0;
    int64_t *pb = NULL, *own = NULL;
    float *ps = NULL;
    int *po = NULL;
    /* the lowest held expert of each row until one goes */
    uint8_t *has_low = (uint8_t *)calloc((size_t)m, 1);
    int *low_x = (int *)malloc((size_t)m * sizeof(int));
    float *low_v = (float *)malloc((size_t)m * sizeof(float));
    memset(changed, 0, (size_t)m);
    int stop = nc < 4 * limit ? nc : 4 * limit;
    for (int s = 0; s < stop; ++s) {
        if (n_items >= limit) {
            break;
        }
        int q = order[s], r = ri[q], x = xi[q];
        int64_t slot;
        if (owner != NULL && pool_rows[r] && fpos < nfree) {
            slot = warm + freeb[fpos++];       /* a free block: no victim */
        } else {
            if (!has_low[r]) {
                int u = urow[r];
                if (u >= 0 && !changed[r]) {
                    low_x[r] = va[u];
                    low_v[r] = hlow[u];
                } else {
                    const float *kr = key + (size_t)r * E;
                    const uint8_t *hr = held + (size_t)r * E;
                    float lo = INFINITY;
                    int at = 0;
                    for (int y = 0; y < E; ++y) {
                        float v = hr[y] ? kr[y] : INFINITY;
                        if (v < lo) {
                            lo = v;
                            at = y;
                        }
                    }
                    low_x[r] = at;
                    low_v[r] = lo;
                }
                has_low[r] = 1;
            }
            int victim = low_x[r], vr = r;
            float vs = low_v[r];
            if (owner != NULL && pool_rows[r] && !scanned) {
                scanned = 1;
                for (int64_t b = 0; b < nblocks; ++b) {
                    npb += owner[b] >= 0;
                }
                if (npb > 0) {
                    have_pv = 1;
                    pb = (int64_t *)malloc((size_t)npb * sizeof(int64_t));
                    own = (int64_t *)malloc((size_t)npb * sizeof(int64_t));
                    ps = (float *)malloc((size_t)npb * sizeof(float));
                    po = (int *)malloc((size_t)npb * sizeof(int));
                    int j = 0;
                    for (int64_t b = 0; b < nblocks; ++b) {
                        if (owner[b] >= 0) {
                            pb[j] = b;
                            own[j] = owner[b];
                            ps[j] = incoming[own[j]] ? INFINITY : key[own[j]];
                            ++j;
                        }
                    }
                    hot_argsort(ps, npb, po);
                }
            }
            if (owner != NULL && pool_rows[r] && have_pv) {
                /* or the warm expert of the lowest key of any row (not one
                 * with a copy on the way) */
                int j = -1;
                while (pk < npb) {
                    j = po[pk];
                    if (owner[pb[j]] == own[j] && ps[j] < INFINITY) {
                        break;
                    }
                    ++pk;
                }
                if (pk < npb && ps[j] < vs) {
                    vr = (int)(own[j] / E);
                    victim = (int)(own[j] % E);
                    vs = ps[j];
                }
            }
            if (key[(size_t)r * E + x] <= vs) {
                continue;
            }
            slot = slots[vr][victim];
            slots[vr][victim] = -1;
            held[(size_t)vr * E + victim] = 0;
            has_low[vr] = 0;
            changed[vr] = 1;
        }
        if (slot >= warm) {
            owner[slot - warm] = (int64_t)r * E + x;
        }
        incoming[(size_t)r * E + x] = 1;
        items[3 * n_items] = r;
        items[3 * n_items + 1] = x;
        items[3 * n_items + 2] = slot;
        ++n_items;
    }
    free(ord_r); free(ri); free(xi); free(inv); free(ur); free(hlow); free(va); free(urow);
    free(gain); free(order); free(freeb); free(has_low); free(low_x); free(low_v);
    free(pb); free(own); free(ps); free(po);
    return n_items;
}
