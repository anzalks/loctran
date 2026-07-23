import pytest
from loctran.extract import merge_words

def test_merge_words_font_size():
    word_list = [
        {"text": "Hello", "x0": 10, "x1": 50, "top": 10, "bottom": 20, "height": 10, "size": 12.0},
        {"text": "world", "x0": 55, "x1": 100, "top": 10, "bottom": 20, "height": 10, "size": 12.0},
        # outlier superscript
        {"text": "1", "x0": 102, "x1": 110, "top": 5, "bottom": 12, "height": 7, "size": 8.0},
    ]
    sx = 2.0
    sy = 2.0
    img_h = 1000.0

    seg = merge_words(word_list, sx, sy, img_h)
    
    # 12.0 * 2.0 = 24.0. The median of [16.0, 24.0, 24.0] is 24.0
    assert seg["font_px"] == 24.0
    
def test_merge_words_font_size_fallback():
    word_list = [
        {"text": "Hello", "x0": 10, "x1": 50, "top": 10, "bottom": 20, "height": 10}, # no size
        {"text": "world", "x0": 55, "x1": 100, "top": 10, "bottom": 20, "height": 10, "size": -5.0}, # invalid size
    ]
    sx = 2.0
    sy = 2.0
    img_h = 1000.0

    seg = merge_words(word_list, sx, sy, img_h)
    assert "font_px" not in seg

def test_merge_words_font_size_out_of_bounds():
    word_list = [
        {"text": "Huge", "x0": 10, "x1": 50, "top": 10, "bottom": 20, "height": 10, "size": 300.0},
    ]
    sx = 2.0
    sy = 2.0
    img_h = 1000.0
    # 300 * 2 = 600, which is > 0.5 * 1000 (500)
    seg = merge_words(word_list, sx, sy, img_h)
    assert "font_px" not in seg
