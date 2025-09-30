def recursive_key_update(d, key_update_func):
    if not isinstance(d, dict) and not isinstance(d, list):
        return d

    if isinstance(d, list):
        for item in d:
            recursive_key_update(item, key_update_func)
        return d

    updated_key_pairs = []
    for key, _ in d.items():
        updated_key = key_update_func(key)
        if key != updated_key:
            updated_key_pairs.append((key, updated_key))

    for key_pair in updated_key_pairs:
        old_key, updated_key = key_pair
        d[updated_key] = d[old_key]
        del d[old_key]

    for _, value in d.items():
        recursive_key_update(value, key_update_func)
    return d
