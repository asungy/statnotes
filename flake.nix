{
  inputs = {
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    nixpkgs-stable.url = "github:nixos/nixpkgs/release-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem(system:
      let
        overlays = [];
        pkgs-stable = import inputs.nixpkgs-stable { inherit system overlays; };
        pkgs-unstable = import inputs.nixpkgs-unstable { inherit system overlays; };
      in
      {
        devShells.default = pkgs-unstable.mkShell {
          packages = with pkgs-unstable; [
            uv
          ];

          shellHook = ''
            alias create-venv='uv venv'
            alias activate='source ./.venv/bin/activate'
            alias venv-install='uv pip install -r <(uv pip compile pyproject.toml)'
            alias run-marimo='uv run marimo edit --watch main.py'
          '';
        };
      }
    );
}
