# Docker source releases

- Deliver the small Docker source build bundle. The recipient builds the image
  and downloads dependencies and model weights on their own server.
- Keep only the latest published Docker source release on GitHub. After a new
  release is published and its asset sizes and SHA-256 digests are verified,
  delete superseded `llama8b-source-*` releases, their assets, and their matching
  version tags. The user has authorized this cleanup after each new release.
- Preserve unrelated releases, Git commit history, and training data. Prefer
  the repository's `/releases/latest` link when sharing the download location.
