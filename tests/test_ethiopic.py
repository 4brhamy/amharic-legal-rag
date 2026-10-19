import pytest

from amlegal.ethiopic import ethiopic_to_int, int_to_ethiopic, normalize, tokenize


@pytest.mark.parametrize("numeral,value", [
    ("፩", 1), ("፱", 9), ("፲", 10), ("፲፩", 11), ("፳፫", 23), ("፺፱", 99),
    ("፻", 100), ("፻፩", 101), ("፻፳", 120), ("፪፻", 200), ("፫፻፳፫", 323),
])
def test_ethiopic_numerals(numeral, value):
    assert ethiopic_to_int(numeral) == value
    assert int_to_ethiopic(value) == numeral


def test_round_trip_all_article_numbers():
    for n in range(1, 400):
        assert ethiopic_to_int(int_to_ethiopic(n)) == n


@pytest.mark.parametrize("bad", ["", "abc", "፩x"])
def test_invalid_numerals_raise(bad):
    with pytest.raises(ValueError):
        ethiopic_to_int(bad)


def test_normalize_unifies_homophones_and_punctuation():
    # ሐ/ኀ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ; Ethiopic punctuation becomes spaces
    assert normalize("ሕግ") == normalize("ህግ")
    assert normalize("ሠላም") == normalize("ሰላም")
    assert normalize("ዐመት") == normalize("አመት")
    assert normalize("ፀሐይ") == normalize("ጸሀይ")
    assert normalize("ጋብቻ፣ ፍቺ።") == "ጋብቻ ፍቺ"


def test_tokenize():
    assert tokenize("የጋብቻ ዕድሜ ስንት ነው?") == ["የጋብቻ", "እድሜ", "ስንት", "ነው"]
