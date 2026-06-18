# Release

twopy publishes to PyPI through GitHub Actions Trusted Publishing. No PyPI API token secret is required.

## One-Time Setup

1. Add a trusted publisher for the `twopy` project in PyPI.
2. Use owner `gumadeiras`, repository `twopy`, workflow `publish-to-pypi.yml`, and environment `pypi`.
3. In GitHub, create the `pypi` environment and require manual approval.

## Release Flow

1. Update `project.version` in `pyproject.toml`.
2. Move the current `CHANGELOG.md` `Unreleased` notes into a new version section with the release date.
3. Add a fresh empty `Unreleased` section at the top of `CHANGELOG.md` for future changes.
4. Run `micromamba run -n twopy pre-commit run --all-files`.
5. Commit the version and changelog changes.
6. Create a GitHub release whose tag is the same version, with or without a leading `v`.
7. Publish the release.
8. Approve the `pypi` deployment.

Use the full pushed commit SHA when creating the release. Short SHAs can be rejected by the GitHub release API.

```sh
version="<release-version>"
sha="$(git rev-parse HEAD)"
git push origin HEAD:refs/heads/main

gh release create "$version" \
  --target "$sha" \
  --title "$version" \
  --notes-file RELEASE_NOTES.md
```

If the publish workflow waits for the `pypi` environment, approve the pending deployment with the numeric environment id returned by GitHub. The API needs raw JSON so the id is sent as an integer.

```sh
gh run list --workflow publish-to-pypi.yml --limit 5
gh api repos/gumadeiras/twopy/actions/runs/<run_id>/pending_deployments

gh api --method POST \
  repos/gumadeiras/twopy/actions/runs/<run_id>/pending_deployments \
  --input - <<'EOF'
{"environment_ids":[<environment_id>],"state":"approved","comment":"Release <version>"}
EOF
```

After the workflow succeeds, PyPI JSON can lag briefly. Verify the public version after a short wait.

```sh
python - <<'PY'
import json
from urllib.request import urlopen

with urlopen("https://pypi.org/pypi/twopy/json", timeout=20) as response:
    data = json.load(response)
print(data["info"]["version"])
PY
```

## Changelog Rules

- Every release must update `CHANGELOG.md` before the release tag is created.
- `CHANGELOG.md` must always keep an `Unreleased` section at the top for future entries.
- New user-facing changes should be added to `Unreleased` as they land.
- Use user-facing language whenever possible. Describe what changed for people using twopy, not repository maintenance.
- Use these sections when they apply: `Features`, `Fixes`, and `Changes`.
- Omit empty sections.
- Do not include release chores such as README screenshots, release guide moves, publishing mechanics, or package-upload status unless the change affects how users install or use twopy.

The release workflow checks that the tag matches `pyproject.toml`, builds the wheel and source distribution, checks package metadata, and publishes to PyPI after the `pypi` environment approval.
