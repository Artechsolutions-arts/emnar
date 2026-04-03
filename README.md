# Emnar — Setup Guide

This guide gets Emnar running from scratch on a new machine. Follow it top to bottom without skipping steps.

---

## What is this?

Emnar is a private, on-premises AI assistant built on top of the [Onyx](https://github.com/onyx-dot-app/onyx) platform. It runs entirely in Docker — no cloud required except for the LLM API key.

**Repository layout:**

```
emnar/          Our custom code (branding, migrations, pipeline, patches)
onyx/           Upstream Onyx platform (treat as vendor — do not edit directly)
```

---

## Prerequisites

Install all of these before you start.

| Tool | Minimum version | Download |
|------|----------------|----------|
| Git | any | https://git-scm.com |
| Docker Desktop | 4.x | https://www.docker.com/products/docker-desktop |
| Docker Compose | v2.x (bundled with Docker Desktop) | included above |

**Docker Desktop settings to change before continuing:**

1. Open Docker Desktop → Settings → Resources
2. Set Memory to at least **8 GB** (16 GB recommended)
3. Set CPUs to at least **4**
4. Apply & Restart

Verify everything works:

```bash
docker --version          # Docker version 24.x or higher
docker compose version    # Docker Compose version v2.x or higher
```

---

## Step 1 — Clone the repository

```bash
git clone <your-repo-url> emnar
cd emnar
```

> Replace `<your-repo-url>` with the actual Git remote URL.

---

## Step 2 — Create the environment file

The `.env` file holds all configuration. Create it from the template:

```bash
cp onyx/deployment/docker_compose/env.template onyx/deployment/docker_compose/.env
```

Now open `onyx/deployment/docker_compose/.env` in any text editor and set these values:

### Required — set these or nothing works

```env
# A random secret for signing auth tokens.
# Generate one by running:  openssl rand -hex 32
# On Windows (PowerShell): -join ((1..32) | ForEach-Object { '{0:x2}' -f (Get-Random -Max 256) })
USER_AUTH_SECRET=replace_this_with_a_random_string

# Your OpenAI API key (or other LLM provider key — see Optional section below)
GEN_AI_API_KEY=sk-...your-key-here...
```

### Recommended — change these for security

```env
# Database password
POSTGRES_PASSWORD=change_this_password

# MinIO credentials (S3-compatible file storage)
S3_AWS_ACCESS_KEY_ID=change_this_access_key
S3_AWS_SECRET_ACCESS_KEY=change_this_secret_key
MINIO_ROOT_USER=change_this_access_key
MINIO_ROOT_PASSWORD=change_this_secret_key
```

> **Important:** `S3_AWS_ACCESS_KEY_ID` and `MINIO_ROOT_USER` must be the same value.
> Same for `S3_AWS_SECRET_ACCESS_KEY` and `MINIO_ROOT_PASSWORD`.

### Leave these as-is (they work out of the box)

```env
IMAGE_TAG=latest
AUTH_TYPE=basic
ENABLE_PAID_ENTERPRISE_EDITION_FEATURES=false
POSTGRES_USER=postgres
POSTGRES_HOST=relational_db
COMPOSE_PROFILES=s3-filestore
FILE_STORE_BACKEND=s3
S3_ENDPOINT_URL=http://minio:9000
OPENSEARCH_FOR_ONYX_ENABLED=true
VESPA_HOST=index
REDIS_HOST=cache
MODEL_SERVER_HOST=inference_model_server
INDEXING_MODEL_SERVER_HOST=indexing_model_server
INTERNAL_URL=http://api_server:8080
WEB_DOMAIN=http://localhost
LOG_ONYX_MODEL_INTERACTIONS=False
```

---

## Step 3 — Build the frontend image

The web frontend must be built from source (it contains Emnar branding). This step takes **10–20 minutes** on first run.

```bash
cd onyx/deployment/docker_compose
docker compose build web_server
```

You will see a lot of output. It is done when you see something like:

```
=> exporting to image
=> => writing image sha256:...
=> => naming to docker.io/onyxdotapp/onyx-web-server:latest
```

If you see `error` in red text, scroll up to find the first error and fix it before continuing.

---

## Step 4 — Start all services

From the `onyx/deployment/docker_compose/` directory:

```bash
docker compose up -d
```

This starts all services in the background. First run will pull several Docker images (~5–10 GB total) and may take 10–30 minutes depending on your connection.

Check that everything started:

```bash
docker compose ps
```

You should see all services with status `running` or `healthy`. If any show `exited`, check that service's logs:

```bash
docker compose logs <service-name>
# Example:
docker compose logs api_server
docker compose logs web_server
```

---

## Step 5 — Open Emnar

Once all services are running, open your browser:

```
http://localhost
```

You should see the Emnar login page. Create your first admin account and log in.

> If the page doesn't load after 2 minutes, check `docker compose logs nginx` for errors.

---

## Stopping and starting

```bash
# Stop everything (data is preserved)
docker compose down

# Start again
docker compose up -d

# Stop and delete ALL data (destructive — cannot be undone)
docker compose down -v
```

---

## Updating Emnar

When there is a new version of the code:

```bash
git pull

# Rebuild the frontend (required after any web source change)
cd onyx/deployment/docker_compose
docker compose build web_server

# Restart with the new image
docker compose up -d
```

---

## Services overview

| Service | What it does | Internal port |
|---------|-------------|---------------|
| `nginx` | Reverse proxy — the only thing exposed to your browser | 80 |
| `web_server` | Next.js frontend (Emnar UI) | 3000 |
| `api_server` | FastAPI backend | 8080 |
| `background` | Celery workers for indexing and connectors | — |
| `relational_db` | PostgreSQL database | 5432 |
| `index` | Vespa search engine | 19071 |
| `opensearch` | Hybrid full-text search | 9200 |
| `inference_model_server` | LLM inference | 9000 |
| `indexing_model_server` | Document embedding | 9000 |
| `cache` | Redis (session/task queue) | 6379 |
| `minio` | S3-compatible file storage | 9000 |

All services communicate on an internal Docker network. Only `nginx` (port 80) is reachable from your browser.

---

## Using a different LLM provider

By default Emnar uses OpenAI. To use a different provider, set these in your `.env`:

**Azure OpenAI:**
```env
GEN_AI_MODEL_PROVIDER=azure
GEN_AI_API_KEY=your-azure-key
GEN_AI_API_ENDPOINT=https://your-resource.openai.azure.com/
GEN_AI_API_VERSION=2024-02-01
GEN_AI_MODEL_VERSION=gpt-4o
```

**Anthropic Claude:**
```env
GEN_AI_MODEL_PROVIDER=anthropic
GEN_AI_API_KEY=sk-ant-...
GEN_AI_MODEL_VERSION=claude-3-5-sonnet-20241022
```

After changing `.env`, restart the affected services:

```bash
docker compose up -d api_server background
```

---

## Custom OCR + Embedding pipeline

Emnar includes a document ingestion pipeline that does OCR on PDFs and creates vector embeddings for search.

### Windows (CUDA GPU required)

```bash
cd emnar/pipeline

# Install dependencies (do this once)
pip install -r requirements_windows.txt

# Run on a PDF
python ocr_embed_pipeline.py path/to/document.pdf

# Options
python ocr_embed_pipeline.py path/to/document.pdf --force-ocr   # re-OCR even if cached
python ocr_embed_pipeline.py path/to/document.pdf --skip-embed  # OCR only, no embedding
python ocr_embed_pipeline.py path/to/document.pdf --pages 5     # limit to first 5 pages
```

### Mac (Apple Silicon)

```bash
cd emnar/pipeline
pip install -r requirements_mac.txt
python ocr_embed_pipeline_mac.py path/to/document.pdf
```

The pipeline requires the PostgreSQL database to be running (start the Docker stack first).

---

## Troubleshooting

### "Connection refused" when opening localhost

Services take 1–3 minutes to become healthy after `docker compose up`. Wait and retry.

Check service health:
```bash
docker compose ps
```

### "web_server" shows "exited"

View logs:
```bash
docker compose logs web_server
```

If you see TypeScript errors, the frontend build failed. Re-run Step 3.

### "api_server" keeps restarting

Usually means `USER_AUTH_SECRET` or `POSTGRES_PASSWORD` is missing from `.env`.

Check:
```bash
docker compose logs api_server | tail -50
```

### "No LLM configured" message in the UI

You did not set `GEN_AI_API_KEY` in `.env`. Add it, then:
```bash
docker compose up -d api_server
```

### Port 80 already in use

Something else is using port 80. Find and stop it, or change the nginx port in `docker-compose.yml`.

On Windows, IIS often occupies port 80. Disable it:
```
Start → Services → World Wide Web Publishing Service → Stop
```

### Out of disk space during build

Docker images total ~15–20 GB. Free up space:
```bash
docker system prune -f        # remove stopped containers and dangling images
docker volume prune -f        # WARNING: removes unused volumes (no data loss if stack is running)
```

### Reset everything and start fresh

```bash
docker compose down -v        # stops all containers and deletes all volumes (ALL DATA GONE)
docker compose up -d          # fresh start
```

---

## Directory reference

```
emnar/
├── assets/             Logo SVGs
├── backend/
│   ├── migrations/     Custom database schema (departments, docs, embeddings)
│   ├── patches/        Files that override onyx/backend/ equivalents
│   └── scripts/        One-off admin scripts
├── pipeline/           OCR → embed ingestion pipeline
└── tools/              CLI utilities

onyx/
├── backend/            FastAPI app (do not edit)
├── web/                Next.js frontend (do not edit except for approved patches)
└── deployment/
    ├── docker_compose/ ← Work here: .env, docker-compose.yml
    └── data/nginx/     Nginx config and Emnar theme CSS
```

---

## Applying patches after an Onyx upgrade

When upgrading Onyx (pulling a new version of `onyx/`), re-apply Emnar's patches:

```bash
cp emnar/backend/patches/auth/schemas.py onyx/backend/onyx/auth/schemas.py
```

Then rebuild the backend and restart:

```bash
docker compose build api_server background
docker compose up -d
```
