# Deployment Guide

This project currently supports two deployment layers:

1. CI and release automation through GitHub Actions
2. a documented hosted deployment target for the demo dashboard

The current GitHub workflow in [.github/workflows/deploy.yml](/Users/swethachakravarthy/Projects/incident-intelligence/.github/workflows/deploy.yml) publishes versioned container images to GitHub Container Registry. That is the release artifact step.

## Recommended Hosted Target

The simplest hosted target for this repository is:

- API: Render Web Service using [Dockerfile.api](/Users/swethachakravarthy/Projects/incident-intelligence/Dockerfile.api)
- Frontend: Render Static Site using [web/package.json](/Users/swethachakravarthy/Projects/incident-intelligence/web/package.json)

This split works well because:

- the backend already runs as a standalone FastAPI service
- the frontend is a static Vite build
- the dashboard can point to a hosted API URL through `VITE_API_BASE_URL`

## Release Flow

The intended release flow is:

1. Open a pull request and let CI pass.
2. Merge to `main`.
3. GitHub Actions publishes fresh API and web images to GHCR.
4. Render auto-deploys the backend from `main` or from the published image strategy you choose.
5. Render rebuilds the frontend static site and points it at the deployed API base URL.

For interview/demo purposes, this gives you a clear story:

- CI validates tests and builds
- GitHub Actions produces release artifacts
- Render hosts the live demo

## Render Setup

### Backend API service

Create a Render Web Service with:

- Environment: `Docker`
- Root directory: repo root
- Dockerfile path: `Dockerfile.api`
- Health check path: `/api/health`

Suggested environment variables:

- `API_HOST=0.0.0.0`
- `API_PORT=8000`
- `API_CORS_ORIGINS=https://<your-frontend-domain>`
- `MPLBACKEND=Agg`
- `MPLCONFIGDIR=/tmp`

### Frontend static site

Create a Render Static Site with:

- Root directory: `web`
- Build command: `npm ci && npm run build`
- Publish directory: `dist`

Required environment variable:

- `VITE_API_BASE_URL=https://<your-api-domain>`

## Deployment Smoke Test

After deploying:

1. Open the frontend URL.
2. Verify the API health endpoint responds at `<api-url>/api/health`.
3. Confirm dashboard summary data loads.
4. Switch between snapshot and temporal views.
5. Start a pipeline run from the dashboard and confirm the job list/log updates.

## Important Limitation

The repository does not yet contain a provider-specific auto-deploy manifest such as a Render blueprint, Fly config, or Cloud Run service definition. The release automation is complete, but the last-mile hosted deployment is currently documented rather than fully automated.
