#!/usr/bin/env sh
set -e

# Script to generate pylock.toml from scratch
# If pylock.toml already exists just run `pdm lock --update-reuse`

# Check if pdm is available
if ! command -v pdm >/dev/null 2>&1
then
    echo "This script requires 'pdm' but it's not installed."
    exit 1
fi

# Locking all dependencies to the same version for all supported
# python versions is not possible (mostly due to numpy)
# so we need to lock separately for python >=3.12 and <3.12
pdm lock --python "~=3.12" --update-reuse
pdm lock --append --python "<3.12" --update-reuse
