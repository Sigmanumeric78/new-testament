# GHCR Publish Plan (Phase 09F)

## Workflows

- `backend-ci.yml`
  - Runs on pull requests and pushes to `main`.
  - Installs backend dependencies.
  - Runs targeted backend contract tests.

- `frontend-ci.yml`
  - Runs on pull requests and pushes to `main`.
  - Uses Node 20 with npm cache.
  - Runs `npm ci`, `npm run lint`, `npm run test:run`, `npm run build` in `frontend/`.

- `docker-publish.yml`
  - Runs on:
    - pushes to `main`
    - tag pushes matching `v*`
    - manual `workflow_dispatch`
  - Builds backend Docker image from `backend/Dockerfile`.
  - Publishes to GHCR.

## GHCR Image Name

- `ghcr.io/sigmanumeric78/new-testament-api`

## Tag Strategy

- `sha-<GITHUB_SHA>` on every docker-publish run.
- `latest` when branch is `main`.
- `<version-tag>` when Git ref is a tag like `v0.1.0`.

## Required GitHub Repository Settings

- Actions permissions should allow:
  - `contents: read`
  - `packages: write`

## Secrets

- No custom repository secrets are required for GHCR publish.
- Workflow uses built-in `GITHUB_TOKEN` via `docker/login-action`.

## Future Azure Container Apps Usage

- Future infrastructure phase will configure Azure Container Apps to pull:
  - `ghcr.io/sigmanumeric78/new-testament-api:<tag>`
- This phase does not perform Azure deployment.
