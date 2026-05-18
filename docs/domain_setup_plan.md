# Domain Setup Plan

## Context
- GitHub Student Pack includes a Namecheap `.me` domain offer.
- Backend target is Azure Container Apps.

## Suggested Domain Strategy
- `api.<domain>.me` -> backend API on Azure Container Apps
- `app.<domain>.me` or `www.<domain>.me` -> frontend app
- apex/root (`<domain>.me`) can redirect to frontend later

## Azure Container Apps Domain Notes
- Subdomains usually use a `CNAME` to Azure-generated app domain.
- Azure verification generally requires a `TXT` record like `asuid.<subdomain>`.
- Apex/root can require `A` records plus TXT verification.

## Timing and Security
- Do not configure final DNS until Azure app endpoint exists.
- Do not expose backend without HTTPS/TLS.
