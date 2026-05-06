# Release

twopy publishes to PyPI through GitHub Actions Trusted Publishing. No PyPI API
token secret is required.

## One-Time Setup

1. Add a trusted publisher for the `twopy` project in PyPI.
2. Use owner `gumadeiras`, repository `twopy`, workflow
   `publish-to-pypi.yml`, and environment `pypi`.
3. In GitHub, create the `pypi` environment and require manual approval.

## Release Flow

1. Update `project.version` in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Run `micromamba run -n twopy pre-commit run --all-files`.
4. Commit the version and changelog changes.
5. Create a GitHub release whose tag is the same version, with or without a
   leading `v`.
6. Publish the release.
7. Approve the `pypi` deployment.

## Changelog Rules

- Every release must update `CHANGELOG.md` before the release tag is created.
- Use user-facing language whenever possible. Describe what changed for people
  using twopy, not repository maintenance.
- Use these sections when they apply: `Features`, `Fixes`, and `Changes`.
- Omit empty sections.
- Do not include release chores such as README screenshots, release guide moves,
  publishing mechanics, or package-upload status unless the change affects how
  users install or use twopy.

The release workflow checks that the tag matches `pyproject.toml`, builds the
wheel and source distribution, checks package metadata, and publishes to PyPI
after the `pypi` environment approval.
