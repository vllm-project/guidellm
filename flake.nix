{
  description = "Tooling and Environments for GuideLLM";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-unstable";
    blueprint.url = "github:numtide/blueprint";
    blueprint.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs:
    inputs.blueprint {
      inherit inputs;
      systems = [ "aarch64-linux" "x86_64-linux" "aarch64-darwin" ];
      prefix = "nix/";
    };
}
