{ pkgs }:
let
  python = pkgs.python312;
in
pkgs.writeShellApplication {
  name = "guidellm-build";
  runtimeInputs = [
    pkgs.git
    pkgs.uv
  ];
  text = ''
    root=$(git rev-parse --show-toplevel)
    export UV_PYTHON=${pkgs.lib.escapeShellArg python.interpreter}
    export UV_PYTHON_DOWNLOADS=never
    # Keep the build-only environment separate from the developer's .venv.
    export UV_PROJECT_ENVIRONMENT="$root/.cache/nix/build-venv"
    unset UV_NO_SYNC UV_FROZEN

    # --only-group excludes GuideLLM and its runtime dependencies. Wheel-only
    # installation prevents dependency builds from fetching unlocked backends.
    # Build both artifacts from the checkout so Git version metadata agrees.
    exec uv --directory "$root" run --locked --only-group build --no-build \
      python -m build --no-isolation --sdist --wheel "$@"
  '';
}
