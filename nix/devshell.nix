{ pkgs, perSystem }:
let
  python = perSystem.self.python;
  nativeLibraries = pkgs.lib.makeLibraryPath (
    [ pkgs.ffmpeg ]
    ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ]
  );
in
pkgs.mkShell {
  packages = [
    python
    pkgs.uv
    pkgs.git
    pkgs.ffmpeg
    pkgs.nodejs
    pkgs.pkg-config
  ];

  env = {
    UV_PYTHON = python.interpreter;
    UV_PYTHON_DOWNLOADS = "never";
    # tox-uv otherwise prefers the UV wheel bundled with its Python package.
    TOX_UV_PATH = "${pkgs.uv}/bin/uv";
  };

  shellHook =
    ''
      unset PYTHONPATH
    ''
    + pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
      export LD_LIBRARY_PATH="${nativeLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ''
    + pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin ''
      export DYLD_LIBRARY_PATH="${nativeLibraries}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    '';
}
