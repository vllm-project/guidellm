{
  pkgs,
  perSystem,
}:
let
  python = pkgs.python312;
  nativeLibraries = pkgs.lib.makeLibraryPath (
    [ pkgs.ffmpeg ]
    ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ]
  );
in
pkgs.mkShellNoCC {
  packages = [
    python
    pkgs.uv
    pkgs.git
    pkgs.ffmpeg
    pkgs.nodejs
    perSystem.self.build
  ];

  env = {
    UV_PYTHON = python.interpreter;
    UV_PYTHON_DOWNLOADS = "never";
    UV_LOCKED = "1";
    # tox-uv otherwise prefers the UV wheel bundled with its Python package.
    TOX_UV_PATH = "${pkgs.uv}/bin/uv";
  };

  shellHook =
    pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
      export LD_LIBRARY_PATH="${nativeLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ''
    + pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin ''
      export DYLD_LIBRARY_PATH="${nativeLibraries}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    '';
}
