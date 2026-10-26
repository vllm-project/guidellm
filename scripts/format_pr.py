#!/usr/bin/env python3
"""Format a PR description by appending a summary of commits.

The current PR body is read from stdin. The output (written to stdout)
cleans up the content above the marker comment and replaces everything
below it with a generated commits section and de-duplicated trailers.

When the PR has a single commit, the git log section is skipped since
the PR body already describes that commit; the trailers are kept.

Usage::

    gh pr view 42 --json body -q .body \\
        | ./scripts/format_pr.py origin/main abc1234 \\
        | gh pr edit 42 --body-file -

Arguments:
    base-ref: Git ref for the PR base branch (e.g. origin/main)
    head-sha: Commit SHA of the PR head
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

MARKER = "<!-- begin:squash-data -->"
SIGNOFF_PREFIX = "Signed-off-by:"


def run_git(*args: str, stdin: str | None = None) -> str:
    """Run a git command and return its stdout without trailing newlines.

    The process exits with git's error message if the command fails.

    :param args: The git command and its arguments.
    :param stdin: Optional text piped into the command.
    :return: The command's standard output.
    """
    result = subprocess.run(
        ["git", *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"error: git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.rstrip("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed namespace with base_ref, head_sha, and cleanup attributes.
    """
    parser = argparse.ArgumentParser(
        description="Format a PR description by appending a summary of commits.",
    )
    parser.add_argument(
        "base_ref",
        help="Git ref for the PR base branch (e.g. origin/main)",
    )
    parser.add_argument(
        "head_sha",
        help="Commit SHA of the PR head",
    )
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean up PR content (default: enabled)",
    )
    return parser.parse_args()


def cleanup_pr_description(body: str) -> str:
    """Apply cleanup rules to the user-authored PR description.

    Rules:
      - Remove Codesmith footer

    :param body: The user-authored portion of the PR body.
    :return: The cleaned description.
    """
    # Remove the codesmith footer
    body = re.sub(
        r"<!-- codesmith:footer -->.*?<!-- /codesmith:footer -->",
        "",
        body,
        flags=re.DOTALL,
    )
    return body.rstrip()


def extract_user_content(body: str) -> str:
    """Extract the user-authored content from the current PR body.

    Everything above the marker line is kept with trailing blank lines
    removed, and everything from the marker onwards is discarded. When
    the body has no marker, the whole body is user content.

    :param body: The current PR body.
    :return: The user-authored portion of the body.
    """
    lines = body.split("\n")
    for index, line in enumerate(lines):
        if MARKER in line:
            lines = lines[:index]
            break
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def list_commit_shas(commit_range: str) -> list[str]:
    """List the non-merge commits in the range, oldest first.

    :param commit_range: Git revision range covering the PR commits.
    :return: The commit SHAs in the range.
    """
    return run_git(
        "log",
        "--reverse",
        "--first-parent",
        "--no-merges",
        "--format=%H",
        commit_range,
    ).split()


def render_commit_log(commit_range: str) -> str:
    """Render the commit log for the range in the medium format.

    :param commit_range: Git revision range covering the PR commits.
    :return: The formatted commit log.
    """
    return run_git(
        "log",
        "--reverse",
        "--first-parent",
        "--no-merges",
        "--format=medium",
        "--no-decorate",
        commit_range,
    )


def unique_trailers(commit_shas: list[str]) -> list[str]:
    """Collect the trailers of all commits, de-duplicated.

    Sign-off trailers keep their original order (with duplicates
    removed) and are listed last; all other trailers are sorted and
    de-duplicated instead.

    :param commit_shas: The commits to collect trailers from.
    :return: The de-duplicated trailer lines.
    """
    trailers: list[str] = []
    for sha in commit_shas:
        message = run_git("log", "-1", "--format=%B", sha)
        parsed = run_git("interpret-trailers", "--parse", stdin=message)
        trailers.extend(line for line in parsed.splitlines() if line.strip())

    other_trailers = sorted(
        {line for line in trailers if not line.startswith(SIGNOFF_PREFIX)}
    )
    signoff_trailers = list(
        dict.fromkeys(line for line in trailers if line.startswith(SIGNOFF_PREFIX))
    )
    return other_trailers + signoff_trailers


def format_pr_description(
    body: str, base_ref: str, head_sha: str, cleanup: bool = True
) -> str:
    """Build the updated PR description from the current body and git data.

    :param body: The current PR body.
    :param base_ref: Git ref for the PR base branch.
    :param head_sha: Commit SHA of the PR head.
    :param cleanup: Apply cleanup rules
    :return: The formatted PR description.
    """
    merge_base = run_git("merge-base", base_ref, head_sha)
    commit_range = f"{merge_base}..{head_sha}"
    commit_shas = list_commit_shas(commit_range)

    user_content = extract_user_content(body)
    if cleanup:
        user_content = cleanup_pr_description(user_content)
    sections = [user_content, MARKER]

    # A single commit is already described by the PR body itself, so the
    # git log is only included when there are multiple commits to squash.
    if len(commit_shas) > 1:
        sections += ["---", "# git log", render_commit_log(commit_range)]

    trailers = unique_trailers(commit_shas)
    if trailers:
        sections += ["---------", "\n".join(trailers)]

    return "\n\n".join(sections) + "\n"


def main() -> None:
    """Read the PR body from stdin and print the formatted description."""
    args = parse_args()
    body = sys.stdin.read().replace("\r", "")
    print(
        format_pr_description(body, args.base_ref, args.head_sha, cleanup=args.cleanup),
        end="",
    )


if __name__ == "__main__":
    main()
