# Releasing allclear

Publishing a GitHub Release (not a draft or prerelease) triggers
`.github/workflows/python-publish.yml`, which uploads to PyPI via Trusted
Publishing. No API token is involved.

1. Bump `version` in `pyproject.toml`, run `uv lock`, and add a `CHANGELOG.md`
   entry. Merge to `main`.
2. `make tag` tags `v<version>` and pushes it.
3. Publish a GitHub Release from that tag.
4. The workflow checks that the tag matches the version and that the version
   isn't already on PyPI, runs the tests, and builds. The upload then waits
   for approval in the `pypi` environment.

PyPI never accepts a re-upload of a version, so a mistake means bumping the
version. `make publish` is a manual fallback that uploads with a token from
`~/.pypirc` and skips the approval step.
