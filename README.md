# Enterprise RAG Ingestion Pipeline

An advanced, production-grade Retrieval-Augmented Generation (RAG) ingestion pipeline designed to process, store, and build high-quality embeddings from uploaded documents. Built with a deeply decoupled, parallelized architecture, it orchestrates complex data processing steps such as document parsing, accurate chunking, Optical Character Recognition (OCR), embeddings generation, and vector indexing.

---

## 🏗 System Architecture

The pipeline orchestrates various enterprise-grade technologies to guarantee fault tolerance, high throughput, and reliable document parsing:

- **FastAPI**: The asynchronous, high-performance web framework serving both the REST APIs (`/api/v1`) and the UI (`/ui`). Middleware validates API Keys (`x-api-key`) to secure ingestion endpoints.
- **PostgreSQL (`pgvector`)**: Serves as the primary persistence layer. It manages Role-Based Access Control (RBAC), stores document/task lifecycle metadata, and efficiently indexes generated embeddings via the `pgvector` extension.
- **Redis**: The unified, rapid state manager. Used to maintain tracking states for long-running asynchronous worker tasks, document workflow status, and task coordination.
- **RabbitMQ**: The message broker that intelligently queues parsing, OCR, and embedding generation jobs. This facilitates decoupled ingestion background processing managed by the `WorkerPool`.
- **SeaweedFS**: A highly scalable, distributed object storage system. Document payloads (PDFs, text files) are securely stored here during transit and ingestion before deep text extraction happens.
- **DotsOCR (Local Pipeline)**: An internal document interpretation suite built around `PyMuPDF`, Tesseract (`pytesseract`), and image conversion flows. Designed to robustly pull data securely behind enterprise firewalls.
- **Embedding / Ranking Engines**: Employs deep NLP models like `sentence-transformers` and `BM25` for generation of dense vectors and semantic matching.

## 🚀 Key Features

* **Multi-Stage Document Parsing:** Seamlessly processes varied text forms and images strictly relying on `DotsOCR`.
* **Reliable Async Processing:** Background `WorkerPool` processes heavy workloads asynchronously across distributed message queues.
* **Service Lifecycle Resilience:** Proactive dependency resolution automatically queries database instances and brokers to ensure application readiness during container bootup.
* **Built-in RBAC:** Out of the box role-based access control, initializing with functional baseline policies (Admin, QA, IT, standard) tailored to business operations.

---

## 🛠 Prerequisites

Make sure you have the following installed on your system:
- **Docker** and **Docker Compose**
- **Python 3.10+** (if running workers or the backend natively)

---

## 📦 Setup and Installation

### 1. Provision Infrastructure
The local environment has been unified using Docker Compose. It will spin up PostgreSQL, Redis, RabbitMQ, and the SeaweedFS master/volume/filer nodes.

```bash
docker-compose up -d
```

Ensure all dependencies pass their `healthcheck` metrics.

### 2. Configure the Environment
Ensure your `.env` file matches the configurations defined in your Compose file and environment requirements:

```ini
PG_USER=<your_postgres_user>
PG_PASSWORD=<your_postgres_password>
PG_DATABASE=<your_postgres_database>
API_KEY=<your_secure_api_key>
# Optional configuration parameters for SeaweedFS & RabbitMQ
```

### 3. Native Application Bootstrapping
Install the target dependencies in your Python environment:
*(Note: Because this utilizes highly optimized subcomponents like DotsOCR, verify PyMuPDF, pytesseract dependencies are accessible locally or built inside your container)*

```bash
pip install -r requirements.txt
```

Launch the FastAPI pipeline server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
This will automatically execute the `lifespan` event loop to verify infrastructure metrics, create database schemas, seed RBAC profiles, and activate the background thread `WorkerPool`.

---

## 🛡 API Security

All major ingestion endpoints under `/api/v1/*` are heavily protected. Include the required header in client requests:

```json
{
  "x-api-key": "<API_KEY>"
}
```
*(By default, `/api/v1/status/health` routes remain exposed for load balancers lacking API authorization payloads)*.

---

## 📁 Repository Structure

```text
├── main.py                # App entrypoint (Dependencies, Lifespan initialization, Workers)
├── src/
│   ├── api/               # Router aggregation, FastAPI endpoint dependencies
│   ├── ingestion/         # Central orchestration, Parsers, Chunking logic
│   ├── database/          # PostgreSQL pool connection, RBAC logic, Redis & Queue definitions
│   ├── integration/ & ocr/ # Interactions with DotsOCR payload pipelines
│   └── service/           # High-level RAG controllers
├── ui/                    # Front-end static bundle assets 
├── docs/ & dots_ocr/      # Internal document manipulation module
└── docker-compose.yml     # Standalone enterprise container spin-up definition
```
