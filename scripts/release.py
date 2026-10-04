#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def replace_once(contents, path, pattern, replacement):
    updated, count = re.subn(pattern, replacement, contents, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one version declaration in {path.relative_to(ROOT)}")
    return updated


def validate_release_artifacts(version):
    dist_dir = ROOT / "tmp" / "dist"
    artifacts = [
        dist_dir / f"tinyopt-{version}-Linux.tar.gz",
        dist_dir / f"tinyopt-{version}-Linux.deb",
        dist_dir / f"tinyopt-{version}.tar.gz",
    ]
    for artifact in artifacts:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty release artifact: {artifact}")

    required_c_api_files = {
        "include/tinyopt/c/c_api.h",
        "include/tinyopt/c/c_api_float.h",
        "include/tinyopt/c/c_api_common.h",
    }
    required_binary_suffixes = {
        "lib/libtinyopt_c.so",
        "include/tinyopt/c/c_api.h",
        "include/tinyopt/c/c_api_float.h",
    }
    with tarfile.open(artifacts[0], "r:gz") as binary_archive:
        binary_files = {
            member.name.removeprefix("./").lstrip("/")
            for member in binary_archive.getmembers()
            if member.isfile()
        }
    missing_binary_files = {
        suffix
        for suffix in required_binary_suffixes
        if not any(path.endswith(suffix) for path in binary_files)
    }
    if missing_binary_files:
        missing = ", ".join(sorted(missing_binary_files))
        raise RuntimeError(f"Binary release archive is missing C API files: {missing}")

    deb_contents = subprocess.run(
        ["dpkg-deb", "--contents", str(artifacts[1])],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    missing_deb_files = {
        suffix for suffix in required_binary_suffixes if suffix not in deb_contents
    }
    if missing_deb_files:
        missing = ", ".join(sorted(missing_deb_files))
        raise RuntimeError(f"Debian release package is missing C API files: {missing}")

    required_source_files = {
        "CMakeLists.txt",
        "README.md",
        "include/tinyopt/tinyopt.h",
        *required_c_api_files,
        "src/opt_dyn.cpp",
        "cmake/GenerateFixedCAPI.cmake",
        "docs/c_api.md",
    }
    with tarfile.open(artifacts[2], "r:gz") as source_archive:
        members = source_archive.getmembers()
        packaged_files = {
            member.name.split("/", 1)[1]
            for member in members
            if member.isfile() and "/" in member.name
        }
    packaged_tmp_entries = [
        member.name
        for member in members
        if "/" in member.name
        and member.name.split("/", 1)[1].removeprefix("./").startswith("tmp/")
    ]
    if packaged_tmp_entries:
        raise RuntimeError(f"Source archive includes temporary files: {packaged_tmp_entries[0]}")

    missing_source_files = required_source_files - packaged_files
    if missing_source_files:
        missing = ", ".join(sorted(missing_source_files))
        raise RuntimeError(f"Source archive is missing required files: {missing}")
    return artifacts


def publish_release(tag, artifacts):
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI (gh) is required to publish release assets")

    release = subprocess.run(
        ["gh", "release", "view", tag], cwd=ROOT, capture_output=True, text=True
    )
    if release.returncode == 0:
        command = ["gh", "release", "upload", tag, *(str(path) for path in artifacts), "--clobber"]
    elif "release not found" in release.stderr.lower():
        command = [
            "gh",
            "release",
            "create",
            tag,
            *(str(path) for path in artifacts),
            "--verify-tag",
            "--generate-notes",
            "--title",
            tag,
        ]
    else:
        raise subprocess.CalledProcessError(
            release.returncode, release.args, release.stdout, release.stderr
        )
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Update Tinyopt's version and build release packages")
    parser.add_argument("version", help="Release version in MAJOR.MINOR.PATCH format")
    args = parser.parse_args()

    match = VERSION_PATTERN.fullmatch(args.version)
    if match is None:
        parser.error("version must use MAJOR.MINOR.PATCH format, for example 0.6.2")
    if not sys.platform.startswith("linux"):
        parser.error("the DEB package requires Linux; run the release task on Linux")
    if shutil.which("gh") is None:
        parser.error("GitHub CLI (gh) is required; install it or use the Pixi release environment")

    major, minor, patch = match.groups()
    tag = f"v{args.version}"
    worktree = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if worktree:
        parser.error("release requires a clean worktree; commit or stash changes first")

    version_file = ROOT / "cmake/Version.cmake"
    original_contents = {version_file: version_file.read_text()}
    cmake_contents = original_contents[version_file]
    for component, value in (("MAJOR", major), ("MINOR", minor), ("PATCH", patch)):
        cmake_contents = replace_once(
            cmake_contents,
            version_file,
            rf"^set\(TINYOPT_VERSION_{component} [0-9]+\)$",
            f"set(TINYOPT_VERSION_{component} {value})",
        )

    pixi_file = ROOT / "pixi.toml"
    pyproject_file = ROOT / "pyproject.toml"
    original_contents[pixi_file] = pixi_file.read_text()
    original_contents[pyproject_file] = pyproject_file.read_text()
    changes = {
        version_file: cmake_contents,
        pixi_file: replace_once(
            original_contents[pixi_file],
            pixi_file,
            r'^version = "[^"]+"$',
            f'version = "{args.version}"',
        ),
        pyproject_file: replace_once(
            original_contents[pyproject_file],
            pyproject_file,
            r'^version = "[^"]+"$',
            f'version = "{args.version}"',
        ),
    }
    version_paths = ["cmake/Version.cmake", "pixi.toml", "pyproject.toml"]
    if any(original_contents[path] != contents for path, contents in changes.items()):
        for path, contents in changes.items():
            path.write_text(contents)
        subprocess.run(["git", "add", *version_paths], cwd=ROOT, check=True)
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"🔧 chore(release): bump version to {args.version}",
                "-m",
                f"Synchronize CMake, Pixi, and Python package versions for release {args.version}.",
            ],
            cwd=ROOT,
            check=True,
        )
    else:
        latest_subject = subprocess.run(
            ["git", "log", "-1", "--format=%s"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        expected_subject = f"🔧 chore(release): bump version to {args.version}"
        if latest_subject != expected_subject:
            parser.error(
                f"version {args.version} is already set but HEAD is not its version-bump commit"
            )

    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    existing_tag = subprocess.run(
        ["git", "tag", "--list", tag], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if existing_tag:
        tag_commit = subprocess.run(
            ["git", "rev-parse", f"{tag}^{{commit}}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tag_commit != head_commit:
            parser.error(f"tag {tag} already exists on a different commit")

    build_dir = ROOT / "build-release"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DTINYOPT_BUILD_TESTS=OFF",
            "-DTINYOPT_BUILD_DOCS=ON",
            "-DTINYOPT_BUILD_PACKAGES=ON",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "tinyopt_c"], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "docs"], check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "package", "deb", "src"],
        check=True,
    )
    artifacts = validate_release_artifacts(args.version)
    if not existing_tag:
        subprocess.run(["git", "tag", tag], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", tag], cwd=ROOT, check=True)
    publish_release(tag, artifacts)
    print(f"Release {args.version} packages created in {ROOT / 'tmp' / 'dist'} and published as {tag}")


if __name__ == "__main__":
    main()