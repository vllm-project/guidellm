{ pkgs }:
pkgs.python312.overrideAttrs (old: {
  # Blueprint exposes passthru.tests as flake checks, but Python's includes helpers.
  passthru = (old.passthru or { }) // { tests = { }; };
})
