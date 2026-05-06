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
2. Run `micromamba run -n twopy pre-commit run --all-files`.
3. Commit the version change.
4. Create a GitHub release whose tag is the same version, with or without a
   leading `v`.
5. Publish the release.
6. Approve the `pypi` deployment.

The release workflow checks that the tag matches `pyproject.toml`, builds the
wheel and source distribution, checks package metadata, and publishes to PyPI
after the `pypi` environment approval.
