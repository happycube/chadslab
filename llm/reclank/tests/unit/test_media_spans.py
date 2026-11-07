"""The soft rows of an image in a block of the dgemma encoder (encoders/media.py)."""
import numpy as np

from reclank.encoders import media as MD


def _span(start, rows):
    return MD.Span(start, np.asarray(rows, np.float32), False, "image", "k")


def test_place_rows_inside():
    x = np.zeros((5, 2), np.float32)
    rows = np.arange(6, dtype=np.float32).reshape(3, 2)
    MD.place_rows(x, [_span(1, rows)])
    assert np.array_equal(x[1:4], rows)
    assert not x[0].any() and not x[4].any()


def test_place_rows_offset():
    x = np.zeros((4, 2), np.float32)
    rows = np.arange(6, dtype=np.float32).reshape(3, 2)
    # the span is at positions 3..5; x holds positions 2..5
    MD.place_rows(x, [_span(3, rows)], offset=2)
    assert np.array_equal(x[1:4], rows)
    assert not x[0].any()


def test_place_rows_clips_both_ends():
    rows = np.arange(10, dtype=np.float32).reshape(5, 2)
    x = np.zeros((3, 2), np.float32)
    # x holds positions 0..2; the span is at 2..6: one row lands
    MD.place_rows(x, [_span(2, rows)])
    assert np.array_equal(x[2], rows[0])
    assert not x[:2].any()
    # x holds positions 1..3; the span is at 0..4: rows 1, 2, 3 land
    y = np.zeros((3, 2), np.float32)
    MD.place_rows(y, [_span(0, rows)], offset=1)
    assert np.array_equal(y, rows[1:4])


def test_place_rows_without_spans():
    x = np.ones((2, 2), np.float32)
    MD.place_rows(x, None)
    assert np.array_equal(x, np.ones((2, 2), np.float32))


def test_limit_positions_inside_one_span():
    pos = np.arange(8)
    sp = MD.Span(2, np.zeros((4, 2), np.float32), True, "image")
    assert MD.limit_positions(pos, [sp]).tolist() == [0, 1, 5, 5, 5, 5, 6, 7]


def test_limit_positions_with_an_offset():
    pos = np.arange(10, 18)
    sp = MD.Span(2, np.zeros((4, 2), np.float32), True, "image")
    assert MD.limit_positions(pos, [sp], start=10).tolist() == [10, 11, 15, 15, 15, 15, 16, 17]


def test_limit_positions_ignores_causal_spans():
    pos = np.arange(6)
    sp = MD.Span(1, np.zeros((3, 2), np.float32), False, "image")
    assert MD.limit_positions(pos, [sp]).tolist() == list(range(6))


def test_limit_positions_without_spans():
    pos = np.arange(4)
    assert MD.limit_positions(pos, None).tolist() == [0, 1, 2, 3]


def test_limit_positions_clips_to_the_block():
    pos = np.arange(4)
    sp = MD.Span(1, np.zeros((6, 2), np.float32), True, "image")
    # the span ends at 7, past this block (the last key is 3)
    assert MD.limit_positions(pos, [sp], last=3).tolist() == [0, 3, 3, 3]


def test_block_end_avoids_splitting_a_short_span():
    sp = MD.Span(2, np.zeros((3, 2), np.float32), True, "image")
    assert MD.block_end(0, 4, [sp]) == 2
    assert MD.block_end(2, 4, [sp]) == 6
    assert MD.block_end(6, 4, [sp]) == 10


def test_block_end_keeps_a_long_span_split():
    sp = MD.Span(2, np.zeros((6, 2), np.float32), True, "image")
    assert MD.block_end(0, 4, [sp]) == 4


def test_encoder_blocks_a_later_block_sees_its_image():
    # The prompt at position 100 of the cache; the image (prompt positions 2 to 4) starts the
    # second block. Its rows see the whole image at the image's absolute positions.
    sp = MD.Span(2, np.zeros((3, 2), np.float32), True, "image")
    blocks = MD.encoder_blocks(100, 7, 4, [sp])
    assert [(a, end, rows) for a, end, rows, _ in blocks] == [(0, 2, 4), (2, 6, 4), (6, 7, 4)]
    assert blocks[0][3].tolist() == [100, 101, 101, 101]
    assert blocks[1][3].tolist() == [104, 104, 104, 105]
    assert blocks[2][3].tolist() == [106, 106, 106, 106]


def test_encoder_blocks_a_big_block_for_a_long_image():
    sp = MD.Span(2, np.zeros((6, 2), np.float32), True, "image")
    blocks = MD.encoder_blocks(0, 10, 4, [sp], big=8)
    assert [(a, end, rows) for a, end, rows, _ in blocks] == [(0, 8, 8), (8, 10, 4)]
    assert blocks[0][3].tolist() == [0, 1, 7, 7, 7, 7, 7, 7]
    # without a big block the image is split
    assert [(a, end) for a, end, _, _ in MD.encoder_blocks(0, 10, 4, [sp])] == [(0, 4), (4, 8), (8, 10)]


def test_encoder_blocks_text():
    blocks = MD.encoder_blocks(5, 6, 4, None, big=8)
    assert [(a, end, rows) for a, end, rows, _ in blocks] == [(0, 4, 4), (4, 6, 4)]
    assert blocks[1][3].tolist() == [9, 10, 11, 12]


def test_limit_mask_is_causal_or_the_image_block():
    # Gemma 4's blockwise_overlay: causal, or the same image block.
    t, start, end = 10, 2, 6
    q = np.arange(t)[:, None]
    k = np.arange(t)[None, :]
    causal = k <= q
    block = (q >= start) & (q < end) & (k >= start) & (k < end)
    want = causal | block
    sp = MD.Span(start, np.zeros((end - start, 2), np.float32), True, "image")
    limit = MD.limit_positions(np.arange(t), [sp])
    got = k <= limit[:, None]
    assert np.array_equal(got, want)


def test_limit_sliding_mask_matches_causal_or_block_in_the_window():
    # A sliding layer with an image shorter than the window: the union of the
    # sliding causal mask and the image block is the limit mask.
    t, window, start, end = 12, 8, 3, 7
    q = np.arange(t)[:, None]
    k = np.arange(t)[None, :]
    causal = (k <= q) & (k > q - window)
    block = (q >= start) & (q < end) & (k >= start) & (k < end)
    want = causal | block
    sp = MD.Span(start, np.zeros((end - start, 2), np.float32), True, "image")
    limit = MD.limit_positions(np.arange(t), [sp])
    got = (k <= limit[:, None]) & (k > q - window)
    assert np.array_equal(got, want)
