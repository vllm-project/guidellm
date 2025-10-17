#!/usr/bin/env sh
set -ex

usage() {
    echo "Script to generate pylock.toml from scratch"
    echo "If pylock.toml already exists just run \`pdm lock --update-reuse\`"
    echo "Usage: $0 [-f] [-h]"
    echo "  -f    Force update of all dependencies"
    echo "  -h    Show this help message"
}

# Check if pdm is available
if ! command -v pdm >/dev/null 2>&1
then
    echo "This script requires 'pdm' but it's not installed."
    exit 1
fi

FORCE_REGEN=0
while getopts "fh" opt; do
  case $opt in
    f) FORCE_REGEN=1 ;;
    h) usage
       exit 0 ;;
    *) usage
       exit 1 ;;
  esac
done

set +e
update_stratagy="$([ $FORCE_REGEN -eq 0 ] && echo "--update-reuse")"
set -e

# Locking all dependencies to the same version for all supported
# python versions is not possible (mostly due to numpy)
# so we need to lock separately for python >=3.12 and <3.12
# Only set update-reuse if not forcing regeneration
pdm lock --python "~=3.12" $update_stratagy
pdm lock --append --python "<3.12" $update_stratagy
