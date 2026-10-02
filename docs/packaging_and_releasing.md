# Packaging and Releasing

## Build Packages

Run the package task from the repository root:

```shell
pixi run build-pkg
```

This builds the Debian package and source archive in `build-docs/`. Binary packages include the
Doxygen API reference under `share/doc/tinyopt` and the Sphinx guide under `share/doc/tinyopt/guide`;
the source archive retains the Markdown and Sphinx sources. The Debian package target requires a
Linux CPack installation with DEB generator support.

## Build a Local Conda Package

The recipe in `recipe/meta.yaml` builds Tinyopt as a platform-independent package containing the
headers, CMake package files, Doxygen API reference, and Sphinx guide. Build it locally with:

```shell
pixi run build-conda-pkg
```

This creates a package under `dist/conda/noarch/` and does not upload it. Building the docs requires
the Sphinx and Doxygen dependencies declared by the recipe. When preparing a package for a newer
release, update the recipe version and source archive checksum.

## Create a Release

Run releases from Linux with Git configured and a clean worktree. First commit the release tooling
and any other intended changes. The release task refuses to run when tracked or untracked changes
are present, and it refuses to reuse an existing `vX.Y.Z` tag.

```shell
pixi run release 0.7.2
```

The task performs these steps in order:

1. Updates the major, minor, and patch values in `cmake/Version.cmake`, plus the workspace and Python
	package versions in `pixi.toml` and `pyproject.toml`.
2. Creates a version-bump commit containing those three version files.
3. Builds the generic CPack TGZ, Debian package, and source TGZ under `dist/`.
4. Creates the lightweight `v0.7.2` tag on the version-bump commit and pushes it to `origin`.

The release task pushes the tag but does not update the remote branch ref. If package generation or
tag pushing fails after the version commit, resolve the issue and rerun the same command; it resumes
when `HEAD` is the matching version-bump commit. If a local tag already exists, it must point to that
commit, allowing a retry of the tag push. If a release fails before that commit, restore a clean
worktree before trying again.