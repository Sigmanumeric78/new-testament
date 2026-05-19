# Zer0 G0nd0g0l Frontend

React + TypeScript + Vite frontend for the Alcohol Intelligence backend.

## Safety Disclaimer
- Estimates only.
- Not medical advice.
- Not legal advice.
- Never use this app to decide whether it is safe to drive.

## Prerequisites
- Node.js 18+
- Backend API running at `http://localhost:8000` (or configured base URL)

## Environment
Create a local env file:

```bash
cp .env.example .env
```

Required variable:

- `VITE_API_BASE_URL=http://localhost:8000`

## Install

```bash
npm install
```

## Run (dev)

```bash
npm run dev
```

## Local Full-Stack (recommended)

Terminal 1:

```bash
cd backend
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:

```bash
cd frontend
npm run dev
```

Terminal 3:

```bash
./scripts/fullstack_acceptance_check.sh
```

Recommended when Ollama may be cold:

```bash
PREWARM_OLLAMA=true CURL_TIMEOUT=120 ./scripts/fullstack_acceptance_check.sh
```

Open:

`http://localhost:5173`

If port 5173 is already busy:

```bash
lsof -i :5173
kill <PID>
```

Then restart `npm run dev`.

## Build

```bash
npm run build
```

## Docker Build (Frontend)

Build static production image (Nginx) with backend API base URL baked at build time:

```bash
docker build \
  -f frontend/Dockerfile \
  -t alcohol-intelligence-frontend:local \
  --build-arg VITE_API_BASE_URL=http://localhost:8000 \
  frontend
```

Run:

```bash
docker run --rm \
  --name alcohol-intelligence-frontend-test \
  -p 5173:80 \
  alcohol-intelligence-frontend:local
```

Smoke test:

```bash
curl -I http://localhost:5173
curl http://localhost:5173 | head
```

## Notes
- Ask flow calls `POST /ask`
- Intake flow calls `POST /intake`
- Chemical Explorer list/detail/conformer uses:
  - `GET /chemicals`
  - `GET /chemicals/{compound_id}`
  - `GET /chemicals/{compound_id}/conformer`
- Health badge calls `GET /health`
- Debug drawer only renders when debug mode is enabled and debug payload exists
- App routes:
  - `/` Ask workflow
  - `/explorer` Chemical Explorer
- 3D conformer rendering:
  - Uses 3Dmol.js runtime loading when available
  - Falls back to a clear “3D conformer not available”/engine-unavailable message when no 3D model can be rendered
