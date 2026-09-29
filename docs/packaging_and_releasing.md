# Packaging and Releasing

## Build Packages

Run the package task from the repository root:

```shell
pixi run build-pkg
```

This builds the Debian package and source archive in `build-default/`. The Debian package target
requires a Linux CPack installation with DEB generator support.

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
4. Creates the lightweight `v0.7.2` tag on the version-bump commit.

The task does not push commits or tags to a remote. If package generation fails after the version
commit, resolve the build issue and rerun the same command; it resumes packaging only when `HEAD`
is the matching version-bump commit. If a release fails before that commit, restore a clean worktree
before trying again.