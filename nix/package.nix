{
  inputs,
  pkgs,
  perSystem,
}:
let
  python = perSystem.self.python;
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };
  pythonSet =
    (pkgs.callPackage inputs.pyproject-nix.build.packages {
      inherit python;
    }).overrideScope
      (pkgs.lib.composeManyExtensions [
        inputs.pyproject-build-systems.overlays.wheel
        # The workspace's locked build backends take precedence over the defaults.
        (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
        (_final: prev: {
          guidellm = prev.guidellm.overrideAttrs (old: {
            # setup.py probes Git and uses its existing fallback without a .git directory.
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.gitMinimal ];
            meta = (old.meta or { }) // { mainProgram = "guidellm"; };
          });
        })
      ]);
  util = pkgs.callPackages inputs.pyproject-nix.build.util {
    python3 = python;
  };
in
util.mkApplication {
  venv = pythonSet.mkVirtualEnv "guidellm-env" workspace.deps.default;
  package = pythonSet.guidellm;
}
