# Repository Structure

This repository has two top-level directories that serve completely different purposes.
**Never confuse them.**

---

## `onyx/` — Upstream Platform (do not edit)

The full [Onyx](https://github.com/onyx-dot-app/onyx) open-source enterprise RAG platform.
Treat this as a vendor dependency. Changes here will be overwritten when upgrading Onyx.

```
onyx/
├── backend/          FastAPI app, Celery workers, connectors, LLM integration
├── web/              Next.js frontend (admin panel, chat UI)
├── deployment/       Docker Compose, Helm, Terraform configs
├── cli/              Go-based developer CLI
├── ods/              Onyx developer tooling (Go)
├── desktop/          Tauri desktop app
├── extensions/       Chrome extension
├── examples/         Onyx usage examples
└── widget/           Embeddable chat widget
```

**To upgrade Onyx:** copy the new release over `onyx/`, re-apply patches from `emnar/backend/patches/`.

---

## `emnar/` — Emnar/Virchow Custom Code (our work)

Everything built for Virchow's on-premises pharma RAG system lives here.

```
emnar/
├── backend/
│   ├── migrations/         Custom Alembic DB migrations
│   │   └── 0001_initial_rag_schema.py   pgvector schema: departments, documents,
│   │                                     chunks, embeddings (HNSW 384-dim)
│   ├── patches/            Files that override onyx/backend/onyx/ equivalents
│   │   └── auth/
│   │       └── schemas.py  Adds Company (Virchow/Emnar), Department (QA/Production/
│   │                        Accounts/Sales), UserStatus enums to the user model
│   └── scripts/            One-off admin scripts
│
├── pipeline/               OCR → Chunk → Embed ingestion pipeline
│   ├── ocr_embed_pipeline.py       Windows / CUDA version (DotsOCR + BGE)
│   ├── ocr_embed_pipeline_mac.py   Mac Apple Silicon / MPS version
│   ├── requirements_windows.txt    pip deps for Windows/CUDA
│   ├── requirements_mac.txt        pip deps for Mac/MPS
│   └── test-docs/                  Sample pharma PDFs for pipeline testing
│
├── tools/
│   └── md_chunks_ask.mjs   Node.js CLI: query rag.md_chunks via full-text
│                            search, pipe top-k chunks to OpenAI, print answer
│                            + sources. Usage: node emnar/tools/md_chunks_ask.mjs "question"
│
└── assets/
    ├── virchow_logo.svg            Virchow brand logo (full)
    └── virchow_sidebar_logo.svg    Virchow brand logo (sidebar variant)
```

---

## Root-level files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Claude Code project context |
| `STRUCTURE.md` | This file |
| `.gitignore` | Covers both `onyx/` and `emnar/` |
| `.pre-commit-config.yaml` | Pre-commit hooks (upstream) |

---

## How patches work

`emnar/backend/patches/` mirrors the path structure of `onyx/backend/onyx/`.
When deploying, the patch files are copied over their upstream counterparts:

```bash
cp emnar/backend/patches/auth/schemas.py onyx/backend/onyx/auth/schemas.py
```

This keeps the diff explicit and makes upgrades safe — you know exactly what changed.
