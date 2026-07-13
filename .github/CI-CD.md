# CI/CD

`ci-cd.yml` runs for pushes and pull requests targeting `dev` or `main`.

- CI installs Python 3.12 dependencies, compiles the source tree, runs the complete pytest suite, and verifies the Docker image.
- Pushes to `dev` publish GHCR tags `dev` and `dev-<sha>` in the `development` environment.
- Pushes to `main` publish GHCR tags `main`, `main-<sha>`, and `latest` in the `production` environment.
- Publishing is gated by both the tests and the container build; pull requests never publish packages.

The image name is `ghcr.io/<owner>/<repository>` in lowercase. No custom secret is required to publish: the job uses the repository-scoped `GITHUB_TOKEN` with `packages: write`.

To deploy a published image on a server, set `SP2026_API_IMAGE` in `.env` to the desired immutable SHA tag, run `docker compose -f docker-compose.cloud.yml pull api`, and then run `docker compose -f docker-compose.cloud.yml up -d --no-build api`. Runtime database and LLM credentials must remain outside the image.
