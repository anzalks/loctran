from unittest.mock import patch, MagicMock
from loctran.translate import _is_valid_translation, _translate_chunk


def test_is_valid_translation():
    # Valid translations
    assert _is_valid_translation("Bonjour", "Hello") is True
    assert _is_valid_translation("42", "42") is True
    assert _is_valid_translation("N/A", "N/A") is True
    assert _is_valid_translation("Hi", "Hi") is True

    # Invalid due to empty or whitespace
    assert _is_valid_translation("", "Hello") is False
    assert _is_valid_translation("   ", "Hello") is False

    # Invalid due to no alphanumerics
    assert _is_valid_translation("----", "----") is False
    assert _is_valid_translation("…", "...") is False
    assert _is_valid_translation("***", "***") is False
    assert _is_valid_translation("___", "___") is False

    # Invalid due to runaway generation
    assert _is_valid_translation("a" * 81, "Hello") is False  # 81 > max(80, 8*5=40)
    assert (
        _is_valid_translation("a" * 161, "A" * 20) is False
    )  # 161 > max(80, 8*20=160)
    assert _is_valid_translation("a" * 150, "A" * 20) is True  # 150 < 160

    # Invalid due to obvious scaffolding
    assert _is_valid_translation('"translation": "Bonjour"', "Hello") is False
    assert _is_valid_translation('"original": "Hello"', "Hello") is False
    assert (
        _is_valid_translation('Here is the json: [{"translation": "x"}]', "x") is False
    )
    assert _is_valid_translation('```json\n"Bonjour"\n```', "Hello") is False


@patch("loctran.translate._get_translate_client")
def test_translate_chunk_drops_invalid(mock_client):
    # Setup mock client to return one valid and one invalid translation
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.chat.return_value = {
        "message": {
            "content": '[{"id": 1, "translation": "Bonjour"}, {"id": 2, "translation": "----"}]'
        }
    }

    chunk = [{"id": 1, "text": "Hello"}, {"id": 2, "text": "World"}]

    results = _translate_chunk(chunk, "dummy-model", "fr")

    assert 1 in results
    assert results[1] == "Bonjour"

    # The invalid "----" should NOT be in the results!
    assert 2 not in results


@patch("loctran.translate._translate_single_with_retry")
@patch("loctran.translate._get_translate_client")
def test_repair_partial_batch(mock_client, mock_single):
    # Setup mock client to return only 1 of 2 items
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.chat.return_value = {
        "message": {"content": '[{"id": 1, "translation": "Bonjour"}]'}
    }

    # Mock the repair fallback
    mock_single.return_value = "Monde"

    chunk = [{"id": 1, "text": "Hello"}, {"id": 2, "text": "World"}]

    results = _translate_chunk(chunk, "dummy", "fr")

    # It should have both
    assert results[1] == "Bonjour"
    assert results[2] == "Monde"
    mock_single.assert_called_once_with("World", "fr", "dummy")


@patch("loctran.translate._translate_single_with_retry")
@patch("loctran.translate._get_translate_client")
def test_positional_fallback_safe(mock_client, mock_single):
    # Setup mock client to return 3 items without ID and without original,
    # simulating a dropped item and no way to map except position.
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    mock_instance.chat.return_value = {
        "message": {"content": '["Un", "Deux", "Trois"]'}
    }

    # Mock repair fallback
    mock_single.side_effect = [
        "Un Repair",
        "Deux Repair",
        "Trois Repair",
        "Quatre Repair",
    ]

    chunk = [
        {"id": 1, "text": "One"},
        {"id": 2, "text": "Two"},
        {"id": 3, "text": "Three"},
        {"id": 4, "text": "Four"},
    ]

    results = _translate_chunk(chunk, "dummy", "fr")

    # Because len(data) [3] != len(chunk) [4], positional fallback is DISABLED.
    # Therefore NONE of the batch results are mapped.
    # All 4 items must be repaired via the single-item fallback.
    assert mock_single.call_count == 4
    assert results[1] == "Un Repair"
    assert results[2] == "Deux Repair"
    assert results[3] == "Trois Repair"
    assert results[4] == "Quatre Repair"
