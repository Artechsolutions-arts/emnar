import redis, json, time, logging, functools
from typing import List, Optional, Tuple, Any, Callable, Dict
from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, SESSION_TTL, FILE_TTL, WORKER_HB_TTL, DEDUP_TTL
from src.models.schemas import BatchSession, FileProgress, TERMINAL_STAGES

logger = logging.getLogger(__name__)

class TenantRedis(redis.Redis):
    """Redis wrapper that automatically prefixes keys for tenant isolation."""
    def __init__(self, tenant_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tenant_id = tenant_id

    def _prefixed(self, key: Any) -> Any:
        prefix = f"{self.tenant_id}:"
        if isinstance(key, str):
            return key if key.startswith(prefix) else f"{prefix}{key}"
        return key

    def __getattribute__(self, item: str) -> Any:
        attr = super().__getattribute__(item)
        methods_to_prefix = ["get", "set", "setex", "hset", "hget", "hgetall", "expire", "sadd", "srem", "scard", "exists", "delete", "publish", "pipeline", "scan"]
        if item in methods_to_prefix and callable(attr):
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                if args:
                    args = (self._prefixed(args[0]),) + args[1:]
                if "name" in kwargs:
                    kwargs["name"] = self._prefixed(kwargs["name"])
                return attr(*args, **kwargs)
            return wrapper
        return attr

class RK:
    """Redis Keys Generator"""
    STATS = "rag:stats:global"
    @staticmethod
    def session(sid: str)          -> str: return f"rag:session:{sid}"
    @staticmethod
    def session_files(sid: str)    -> str: return f"rag:session:{sid}:files"
    @staticmethod
    def file(fid: str)             -> str: return f"rag:file:{fid}"
    @staticmethod
    def progress_ch(sid: str)      -> str: return f"rag:progress:{sid}"
    @staticmethod
    def worker_hb(wid: str)        -> str: return f"rag:worker:{wid}:heartbeat"
    @staticmethod
    def rate(dept_id: str)         -> str: return f"rag:rate:{dept_id}"
    @staticmethod
    def dedup(content_hash: str)   -> str: return f"rag:dedup:{content_hash}"
    @staticmethod
    def fence(obj_id: str)         -> str: return f"rag:fence:{obj_id}"
    @staticmethod
    def taskset(fid: str)          -> str: return f"rag:taskset:{fid}"

class RedisStateManager:
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.pool = redis.ConnectionPool(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            password=REDIS_PASSWORD, decode_responses=True,
            max_connections=50, health_check_interval=30
        )
        self.pool_bytes = redis.ConnectionPool(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            password=REDIS_PASSWORD, decode_responses=False,
            max_connections=50, health_check_interval=30
        )
        self.r = TenantRedis(tenant_id, connection_pool=self.pool)
        self.r_bytes = TenantRedis(tenant_id, connection_pool=self.pool_bytes)
        logger.info(f"[Redis] Connected {REDIS_HOST}:{REDIS_PORT}/db{REDIS_DB} (Tenant: {tenant_id})")

    def set_fence(self, obj_id: str, ttl: int = 3600, owner: str = "default") -> bool:
        return self.r.set(RK.fence(obj_id), owner, ex=ttl, nx=True) is True

    def clear_fence(self, obj_id: str):
        self.r.delete(RK.fence(obj_id))

    def is_fenced(self, obj_id: str) -> bool:
        return bool(self.r.exists(RK.fence(obj_id)))

    def add_tasks_to_set(self, file_id: str, task_ids: List[str]):
        if not task_ids: return
        self.r.sadd(RK.taskset(file_id), *task_ids)
        self.r.expire(RK.taskset(file_id), FILE_TTL)

    def complete_task(self, file_id: str, task_id: str):
        self.r.srem(RK.taskset(file_id), task_id)

    def get_remaining_tasks(self, file_id: str) -> int:
        return int(self.r.scard(RK.taskset(file_id)))

    def ping(self) -> bool:
        try: return bool(self.r.ping())
        except redis.RedisError: return False

    def create_session(self, session: BatchSession) -> str:
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(RK.session(session.session_id), mapping=session.to_redis_hash())
        pipe.expire(RK.session(session.session_id), SESSION_TTL)
        pipe.execute()
        return session.session_id

    def get_session(self, session_id: str) -> Optional[BatchSession]:
        h = self.r.hgetall(RK.session(session_id))
        if not h: return None
        return BatchSession.from_redis_hash(h)

    def mark_session_complete(self, session_id: str):
        self.r.hset(RK.session(session_id), "status", "complete")

    def session_summary(self, session_id: str) -> Optional[dict]:
        session = self.get_session(session_id)
        if not session: return None
        files_map = self.r.hgetall(RK.session_files(session_id))
        file_ids  = [v for k, v in files_map.items() if k != "_init"]
        files = []
        done = skipped = errors = in_prog = total_chunks = 0
        for fid in file_ids:
            fp = self.get_file_progress(fid)
            if not fp: continue
            files.append(fp.to_dict())
            total_chunks += fp.chunks
            s = fp.stage
            if s == "done":    done    += 1
            elif s == "skipped": skipped += 1
            elif s == "error":   errors  += 1
            else:                in_prog += 1
        return {
            "session_id":   session_id, "total": session.total,
            "done": done, "skipped": skipped, "errors": errors,
            "in_progress": in_prog, "total_chunks": total_chunks,
            "status": session.status, "created_at": session.created_at, "files": files,
        }

    def list_sessions(self, limit: int = 20) -> List[dict]:
        cursor, sessions = 0, []
        pattern = f"{self.tenant_id}:rag:session:*"
        while True:
            cursor, keys = self.r.scan(cursor, match=pattern, count=100)
            for k in keys:
                if isinstance(k, bytes): k = k.decode()
                clean_key = k[len(self.tenant_id)+1:]
                if ":files" in clean_key: continue
                h = self.r.hgetall(clean_key)
                if h and "session_id" in h: sessions.append(h)
            if cursor == 0: break
        sessions.sort(key=lambda x: float(x.get("created_at", 0)), reverse=True)
        return sessions[:limit]

    def register_file(self, session_id: str, fp: FileProgress):
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(RK.file(fp.file_id), mapping=fp.to_redis_hash())
        pipe.expire(RK.file(fp.file_id), FILE_TTL)
        pipe.hset(RK.session_files(session_id), fp.filename, fp.file_id)
        pipe.execute()

    def get_file_progress(self, file_id: str) -> Optional[FileProgress]:
        h = self.r.hgetall(RK.file(file_id))
        return FileProgress.from_redis_hash(h) if h else None

    def init_job(self, file_id: str, session_id: str, filename: str, metadata: Optional[Dict[str, Any]] = None):
        """Standardized job initialization used by API and Tracker."""
        fp = FileProgress(
            file_id=file_id,
            session_id=session_id,
            filename=filename,
            stage="queued",
            pct=0
        )
        self.register_file(session_id, fp)
        if metadata:
            self.r.hset(RK.file(file_id), mapping={k: str(v) for k, v in metadata.items()})

    def get_job_status(self, file_id: str, session_id: str) -> Optional[dict]:
        """Retrieves raw job status dict."""
        fp = self.get_file_progress(file_id)
        return fp.to_dict() if fp else None

    def set_taskset(self, file_id: str, count: int):
        """Initializes a granular task set for chunk-level tracking."""
        task_ids = [f"chunk:{i}" for i in range(count)]
        self.add_tasks_to_set(file_id, task_ids)

    def update_task_status(self, file_id: str, index: int, status: str):
        """Updates status for a specific sub-task (chunk)."""
        task_id = f"chunk:{index}"
        if status == "completed":
            self.complete_task(file_id, task_id)
        # We can extend this to track individual chunk failures if needed


    def update_stage(self, file_id, session_id, stage, pct, extra=None, publish=True):
        updates = {"stage": stage, "pct": str(pct)}
        if extra: updates.update({k: str(v) if v is not None else "" for k, v in extra.items()})
        if stage == "validating": updates.setdefault("started_at", str(time.time()))
        if stage in TERMINAL_STAGES: updates.setdefault("finished_at", str(time.time()))
        self.r.hset(RK.file(file_id), mapping=updates)
        if publish: self._publish(session_id, file_id)

    def _publish(self, session_id: str, file_id: str):
        fp = self.get_file_progress(file_id)
        if not fp: return
        event = {"type": "file_progress", "data": fp.to_dict()}
        try: self.r.publish(RK.progress_ch(session_id), json.dumps(event))
        except: pass
        if fp.stage in ("done", "skipped", "error"): self._try_complete_session(session_id)

    def _try_complete_session(self, session_id: str):
        summary = self.session_summary(session_id)
        if summary and summary["in_progress"] == 0:
            self.mark_session_complete(session_id)
            event = {"type": "session_complete", "data": summary}
            try: self.r.publish(RK.progress_ch(session_id), json.dumps(event))
            except: pass

    def subscribe_session(self, session_id: str):
        summary = self.session_summary(session_id)
        if summary and summary.get("status") == "complete":
            yield {"type": "session_complete", "data": summary}
            return
        pub = self.r_bytes.pubsub(ignore_subscribe_messages=True)
        ch  = f"{self.tenant_id}:{RK.progress_ch(session_id)}"
        pub.subscribe(ch)
        try:
            deadline = time.time() + SESSION_TTL
            while time.time() < deadline:
                msg = pub.get_message(timeout=30)
                if msg is None: yield {"type": "ping"}; continue
                try:
                    data = json.loads(msg["data"])
                    yield data
                    if data.get("type") == "session_complete": break
                except: continue
        finally:
            try: pub.unsubscribe(ch); pub.close()
            except: pass

    def check_rate_limit(self, dept_id, limit=200, window_s=3600) -> Tuple[bool, int]:
        pipe = self.r.pipeline(transaction=True)
        pipe.incr(RK.rate(dept_id)); pipe.expire(RK.rate(dept_id), window_s)
        res = pipe.execute(); count = res[0]
        return count <= limit, count

    def set_dedup(self, content_hash, doc_id) -> bool:
        return self.r.set(RK.dedup(content_hash), doc_id, nx=True, ex=DEDUP_TTL) is True

    def check_dedup(self, content_hash) -> Optional[str]:
        return self.r.get(RK.dedup(content_hash))

    def incr_stat(self, field, by=1): self.r.hincrby(RK.STATS, field, by)

    def global_stats(self) -> dict:
        raw = self.r.hgetall(RK.STATS) or {}
        return {k: int(raw.get(k, 0)) for k in ["total_processed", "total_failed", "total_skipped"]}

    def worker_heartbeat(self, worker_id: str):
        self.r.setex(RK.worker_hb(worker_id), WORKER_HB_TTL, str(time.time()))

    def active_workers(self) -> List[str]:
        workers, cursor = [], 0
        pattern = f"{self.tenant_id}:rag:worker:*:heartbeat"
        while True:
            cursor, keys = self.r.scan(cursor, match=pattern, count=100)
            for k in keys: workers.append(k.split(":")[2])
            if cursor == 0: break
        return workers

    def dashboard(self) -> dict:
        return {
            "active_workers": len(self.active_workers()),
            "worker_ids": self.active_workers(),
            "global_stats": self.global_stats(),
            "recent_sessions": self.list_sessions(limit=10),
        }

    def flush_all(self, confirm="") -> bool:
        if confirm != "YES_DELETE_ALL": return False
        cursor = 0; pattern = f"{self.tenant_id}:rag:*"
        while True:
            cursor, keys = self.r.scan(cursor, match=pattern, count=500)
            if keys: 
                clean_keys = [k[len(self.tenant_id)+1:] for k in keys]
                self.r.delete(*clean_keys)
            if cursor == 0: break
        return True
