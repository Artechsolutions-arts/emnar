import os, logging, uvicorn, asyncio, time
from typing import Dict, Any, cast
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager

from src.config import cfg
from src.database.postgres.queries import get_pg_connection, create_schema, RBACManager, get_pg_pool, seed_rbac
from src.database.redis.redis_db import RedisStateManager
from src.database.queue.rabbitmq_broker import rabbit_connect, setup_topology
from src.storage.seaweedfs_client import SeaweedFSClient
from src.services.rag_pipeline import RAGPipeline
from src.worker.pool import WorkerPool # Corrected import

from src.api.routes import router # Corrected import
from src.observability import setup_logger, instrument_app

logger = setup_logger("main")

# --- State ---
_pool_pg, _rsm, _mq_conn, _pipeline, _pool, _storage, _ids = None, None, None, None, None, None, {}

async def wait_for_services(retries: int = 15, delay: float = 3.0):
    """On-prem resilience: Wait for Database, Redis, and RabbitMQ to be ready."""
    for i in range(retries):
        try:
            # 1. Postgres
            conn = get_pg_connection()
            conn.close()
            # 2. Redis
            from src.database.redis.redis_db import RedisStateManager
            rsm = RedisStateManager()
            if not rsm.r.ping(): raise Exception("Redis down")
            # 3. Rabbit
            mq_conn = rabbit_connect()
            mq_conn.close()
            logger.info("Service discovery successful - All dependencies UP ✓")
            return True
        except Exception as e:
            logger.warning(f"Waiting for services... (Attempt {i+1}/{retries}): {e}")
            await asyncio.sleep(delay)
    return False

def seed_rbac(rbac: RBACManager):
    """Seed base roles/departments for fresh on-prem installs."""
    ids = {}
    
    # 1. Ensure Default Department exists
    try:
        dept_id = rbac.create_department("Standard", "Generic starting department")
    except:
        cur = rbac.conn.cursor()
        cur.execute("SELECT id FROM departments WHERE name=%s", ("Standard",))
        res = cur.fetchone()
        dept_id = str(res[0]) if res else None

    ids["dept_default"] = dept_id

    # 2. Create System User
    system_email = "system@internal.rag"
    try:
        user_id = rbac.create_user(system_email, "System Master", "no_hash_required", dept_id, True)
        logger.info(f"Created System User: {user_id}")
    except:
        cur = rbac.conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=%s", (system_email,))
        res = cur.fetchone()
        user_id = str(res[0]) if res else None

    ids["user_default"] = user_id

    # 3. Create sample departments and users for the UI
    sample_depts = [
        ("Administration", "Admin department"),
        ("QA", "Quality Assurance"),
        ("Plant", "Plant Operations"),
        ("IT", "Information Technology"),
        ("Finance", "Finance and accounting"),
    ]
    sample_users = [
        ("admin@rag.local", "Admin", "Administration", True),
        ("qa@rag.local", "QA Specialist", "QA", False),
        ("it@rag.local", "IT Specialist", "IT", False),
    ]
    
    dept_map: Dict[str, str] = {"Standard": dept_id}
    for name, desc in sample_depts:
        try:
            did = rbac.create_department(name, desc, None)
            dept_map[name] = did
            logger.info(f"Created dept '{name}': {did}")
        except Exception as e:
            cur = rbac.conn.cursor()
            cur.execute("SELECT id FROM departments WHERE name=%s", (name,))
            res = cur.fetchone()
            if res: dept_map[name] = str(res[0])
            
    for email, uname, dept_name, is_admin in sample_users:
        try:
            did = dept_map.get(dept_name)
            if did:
                uid = rbac.create_user(email, uname, "hashed_pw", did, is_admin)
                logger.info(f"Created user '{uname}': {uid}")
        except Exception as e:
            pass # User likely already exists

    return ids

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool_pg, _rsm, _mq_conn, _pipeline, _pool, _storage, _ids
    
    # 0. On-prem service discovery wait
    logger.info("Starting Service Discovery Life-Check...")
    if not await wait_for_services():
        logger.critical("Dependency wait timeout! Application may start in restricted mode.")
    
    # 1. Database & Schema
    try:
        _pool_pg = get_pg_pool(minconn=1, maxconn=cfg.upload_workers + 5)
        conn = _pool_pg.getconn()
        create_schema(conn)
        _ids = seed_rbac(RBACManager(conn))
        _pool_pg.putconn(conn)
        logger.info(f"RBAC Seed IDs: {_ids}")
    except Exception as e:
        logger.error(f"PostgreSQL initialization failed: {e}")

    # 2. Redis
    try:
        _rsm = RedisStateManager(tenant_id="RagDefault")
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")

    # 3. RabbitMQ
    try:
        _mq_conn = rabbit_connect()
        setup_topology(_mq_conn)
    except Exception as e:
        logger.error(f"RabbitMQ initialization failed: {e}")

    # 4. Storage (SeaweedFS)
    try:
        from src.storage.storage_service import StorageService
        _client = SeaweedFSClient(cfg.seaweedfs_master_url, cfg.seaweedfs_filer_url)
        _storage = StorageService(_client)
        logger.info("SeaweedFS Storage service initialized.")
    except Exception as e:
        logger.error(f"SeaweedFS initialization failed: {e}")

    # 5. Pipeline & Router
    _pipeline = RAGPipeline(_pool_pg, _rsm, storage=_storage)

    # 6. Worker Pool
    try:
        _pool = WorkerPool(_rsm, _pipeline, cfg.upload_workers)
        _pool.start()
        logger.info(f"Worker Pool started with {cfg.upload_workers} threads.")
    except Exception as e:
        logger.error(f"Worker Pool failed to start: {e}")
    
    yield
    
    if _pool: _pool.stop()
    if _mq_conn: _mq_conn.close()
    if _pool_pg: _pool_pg.closeall()

app = FastAPI(title="Enterprise RAG Pipeline", lifespan=lifespan)

from src.api.dependencies import get_pipeline
app.dependency_overrides[get_pipeline] = lambda: _pipeline

instrument_app(app)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── API Key Security Middleware ──────────────────────────────────────────────
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

API_KEY = os.getenv("API_KEY", "dots-rag-707-secure-access")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Allow health check without key for monitoring
    if request.url.path.startswith("/api/v1"):
         if "/api/v1/status/health" not in request.url.path:
             api_key = request.headers.get("x-api-key")
             if api_key != API_KEY:
                 return JSONResponse(
                     status_code=403,
                     content={"detail": "Unauthorized: Invalid or missing API Key"}
                 )
    return await call_next(request)

# Include modular RAG routes
app.include_router(router, prefix="/api/v1")

# Serve UI Static Files
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return FileResponse("ui/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

