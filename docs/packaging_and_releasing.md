# Packaging and Releasing

## Build Packages

Run the package task from the repository root:

```shell
pixi run build-pkg
```

This builds the C shared library, Debian package, and source archive in `tmp/dist/`. Binary packages
include `libtinyopt_c`, its generated and precision-specific C headers, the Doxygen API reference
under `share/doc/tinyopt`, and the Sphinx guide under `share/doc/tinyopt/guide`; the source archive
retains the C API sources and documentation. The Debian package target requires a Linux CPack
installation with DEB generator support.
For CMake consumer examples that select the header-only C++ API or the installed C library, see
[Installation and Usage](installation_and_usage.md).

## Create a Release

Run releases from Linux with Git configured and a clean worktree. First commit the release tooling
and any other intended changes. The release task refuses to run when tracked or untracked changes
are present, and it refuses to reuse an existing `vX.Y.Z` tag.

```shell
pixi run release 0.7.0
```

The task performs these steps in order:

1. Updates the major, minor, and patch values in `cmake/Version.cmake` and updates the workspace
   version in `pixi.toml` and Python package version in `pyproject.toml` to the requested release.
2. Creates a version-bump commit containing those three version files.
3. Builds the generic CPack TGZ, Debian package, and source TGZ under `tmp/dist/`, then verifies
   the source archive contains the expected project files and excludes `tmp/`.
4. Creates the lightweight `v0.7.0` tag on the version-bump commit, pushes it to `origin`, and
   creates or updates the GitHub release with all three package files.

Temporary CTest output, package artifacts, and timing reports are kept under the ignored `tmp/`
directory and are excluded from source packages.

The release task does not update the remote branch ref. It requires the GitHub CLI to be available
and authenticated. If package generation, tag pushing, or asset upload fails after the version
commit, resolve the issue and rerun the same command; it resumes when `HEAD` is the matching
version-bump commit. If a local tag already exists, it must point to that commit. Existing GitHub
releases are updated by replacing matching assets, allowing an upload retry. If a release fails
before that commit, restore a clean worktree before trying again.
