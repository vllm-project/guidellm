import pytest

from guidellm.utils.dict import recursive_key_update


def update_str(string):
  return string + "_updated"

@pytest.mark.smoke
def test_recursive_key_update_updates_keys():
  my_dict = {
    "my_key": {
      "my_nested_key": {
          "my_double_nested_key": "someValue"
      },
      "my_other_nested_key": "someValue"
    },
    "my_other_key": "value"
  }
  my_updated_dict = {
    "my_key_updated": {
      "my_nested_key_updated": {
          "my_double_nested_key_updated": "someValue"
      },
      "my_other_nested_key_updated": "someValue"
    },
    "my_other_key_updated": "value"
  }
  recursive_key_update(my_dict, update_str)
  assert my_dict == my_updated_dict


def truncate_str_to_ten(string):
  return string[:10]


@pytest.mark.smoke
def test_recursive_key_update_leaves_unchanged_keys():
  my_dict = {
    "my_key": {
      "my_nested_key": {
          "my_double_nested_key": "someValue"
      },
      "my_other_nested_key": "someValue"
    },
    "my_other_key": "value"
  }
  my_updated_dict = {
    "my_key": {
      "my_nested_": {
          "my_double_": "someValue"
      },
      "my_other_n": "someValue"
    },
    "my_other_k": "value"
  }
  recursive_key_update(my_dict, truncate_str_to_ten)
  assert my_dict == my_updated_dict


@pytest.mark.smoke
def test_recursive_key_update_updates_dicts_in_list():
  my_dict = {
    "my_key":
      [{ "my_list_item_key_1": "someValue" },
       { "my_list_item_key_2": "someValue" },
       { "my_list_item_key_3": "someValue" }]
  }
  my_updated_dict = {
    "my_key_updated":
      [{ "my_list_item_key_1_updated": "someValue" },
       { "my_list_item_key_2_updated": "someValue" },
       { "my_list_item_key_3_updated": "someValue" }]
  }
  recursive_key_update(my_dict, update_str)
  assert my_dict == my_updated_dict
