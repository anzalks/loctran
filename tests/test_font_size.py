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

def test_hocr_line_sizes(monkeypatch):
    import loctran.extract as ext
    
    mock_hocr = """
    <div class='ocr_page'>
        <span class='ocr_line' title='bbox 10 20 100 40; baseline 0 -5; x_size 34.5; x_descenders 5; x_ascenders 5'>Text</span>
        <span class='ocr_line' title='bbox 100 200 200 220; baseline 0 -5; x_descenders 5; x_ascenders 5'>No x_size here</span>
    </div>
    """
    
    def mock_image_to_pdf_or_hocr(img, extension):
        return mock_hocr
        
    monkeypatch.setattr("pytesseract.image_to_pdf_or_hocr", mock_image_to_pdf_or_hocr)
    
    lines = ext._hocr_line_sizes(None)
    assert len(lines) == 1
    bbox, x_size = lines[0]
    assert bbox == (10, 20, 100, 40)
    assert x_size == 34.5

def test_hocr_assignment():
    import loctran.extract as ext
    
    segments = [
        {"bbox": [10, 20, 90, 20]}, # center-y = 30, x span 10-100
        {"bbox": [10, 50, 90, 20]}, # center-y = 60, x span 10-100
        {"bbox": [10, 100, 90, 20]}, # center-y = 110, no match
    ]
    
    hocr_lines = [
        ((5, 15, 105, 45), 20.0), # Matches seg 1 perfectly
        ((0, 45, 50, 75), 25.0), # Matches seg 2, overlap 40
        ((40, 45, 110, 75), 30.0), # Matches seg 2, overlap 60 (larger horizontal overlap, should win)
    ]
    
    ext._assign_hocr_sizes(segments, hocr_lines)
    
    assert segments[0]["font_px"] == 20.0
    assert segments[1]["font_px"] == 30.0
    assert "font_px" not in segments[2]

def test_renderer_exact_size():
    import loctran.render as rnd
    
    seg = {
        "text": "Hello",
        "bbox": [10, 10, 50, 20],
        "font_px": 20.0
    }
    html = rnd.get_overlay_html(1000, 1000, "img.jpg", [seg])
    # image width is 1000. 20.0 / 1000 * 100 = 2.0000cqw
    assert "font-size: 2.0000cqw;" in html

def test_renderer_fallback_size():
    import loctran.render as rnd
    
    seg = {
        "text": "Hello",
        "bbox": [10, 10, 50, 20],
        # no font_px
    }
    html = rnd.get_overlay_html(1000, 1000, "img.jpg", [seg])
    assert "font-size:" in html
    assert "font-size: 2.0000cqw;" not in html

