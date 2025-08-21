import pytest

from guidellm.utils.text import camelize_str


@pytest.mark.smoke
def test_camelize_str_camelizes_string():
    assert camelize_str("no_longer_snake_case") == "noLongerSnakeCase"


@pytest.mark.smoke
def test_camelize_str_leaves_non_snake_case_text_untouched():
    assert camelize_str("notsnakecase") == "notsnakecase"
