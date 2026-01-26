#!/usr/bin/env sh
set -ex

usage() {
    echo "Script to generate python dependency lock"
    echo "Usage: $0 [-f] [-h]"
    echo "  -f    Force update of all dependencies"
    echo "  -h    Show this help message"
}

# Check if uv is available
if ! command -v uv >/dev/null 2>&1
then
    echo "This script requires 'uv' but it's not installed."
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

if [ $FORCE_REGEN -eq 1 ]; then
    uv lock -U
else
    uv lock
fi
