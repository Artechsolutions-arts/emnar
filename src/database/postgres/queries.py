import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json
from src.config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD, EMBEDDING_DIM, EMBEDDING_MODEL, cfg
from src.database.postgres.vector_store import PGVectorStore
from src.ingestion.indexing.indexer import VectorIndexer
import logging, time

logger = logging.getLogger(__name__)

def get_pg_connection():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD,
    )
    conn.autocommit = True
    return conn

def get_pg_pool(minconn=1, maxconn=10):
    return pool.ThreadedConnectionPool(
        minconn, maxconn,
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD,
    )

def create_schema(conn):
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    except Exception as e:
        logger.warning(f"Failed to create extensions (possibly missing privileges): {e}")


    # T1 departments
    cur.execute("""CREATE TABLE IF NOT EXISTS departments (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        name TEXT NOT NULL UNIQUE, description TEXT,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(), created_by UUID);""")
    # T1 dept_access_grants
    cur.execute("""CREATE TABLE IF NOT EXISTS dept_access_grants (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        granting_dept_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
        receiving_dept_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
        granted_by UUID,
        access_type TEXT NOT NULL DEFAULT 'read' CHECK (access_type IN ('read','full')),
        expires_at TIMESTAMP, created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        UNIQUE (granting_dept_id, receiving_dept_id));""")
    # T2 users
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        email TEXT NOT NULL UNIQUE, name TEXT NOT NULL, password_hash TEXT NOT NULL,
        department_id UUID NOT NULL REFERENCES departments(id) ON DELETE RESTRICT,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        is_super_admin BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(), last_login TIMESTAMP);""")
    # deferred FKs
    for cname, table, col, ref in [
        ("fk_dept_created_by",  "departments",       "created_by", "users(id) ON DELETE SET NULL"),
        ("fk_grant_granted_by", "dept_access_grants","granted_by", "users(id) ON DELETE SET NULL"),
    ]:
        cur.execute(f"""DO $$ BEGIN
          IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints
                         WHERE constraint_name='{cname}') THEN
            ALTER TABLE {table} ADD CONSTRAINT {cname}
              FOREIGN KEY ({col}) REFERENCES {ref};
          END IF; END $$;""")
    # T3 chat
    cur.execute("""CREATE TABLE IF NOT EXISTS chat (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
        title TEXT, model_name TEXT NOT NULL DEFAULT 'gpt-4o',
        temperature NUMERIC(3,2) NOT NULL DEFAULT 0.0,
        rag_enabled BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW());""")
    # T3 messages
    cur.execute("""CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        chat_id UUID NOT NULL REFERENCES chat(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user','assistant')),
        content TEXT NOT NULL, created_at TIMESTAMP NOT NULL DEFAULT NOW());""")
    # T3 user_uploads
    cur.execute("""CREATE TABLE IF NOT EXISTS user_uploads (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        chat_id UUID REFERENCES chat(id) ON DELETE SET NULL,
        department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
        file_name TEXT NOT NULL, file_path TEXT NOT NULL, file_size_bytes BIGINT,
        mime_type TEXT NOT NULL DEFAULT 'application/pdf',
        upload_scope TEXT NOT NULL DEFAULT 'dept' CHECK (upload_scope IN ('chat','dept')),
        embed_enabled BOOLEAN NOT NULL DEFAULT TRUE,
        processing_status TEXT NOT NULL DEFAULT 'pending'
            CHECK (processing_status IN ('pending','processing','completed','failed')),
        created_at TIMESTAMP NOT NULL DEFAULT NOW());""")
    # T3 admin_uploads
    cur.execute("""CREATE TABLE IF NOT EXISTS admin_uploads (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        admin_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
        file_name TEXT NOT NULL, file_path TEXT NOT NULL, file_size_bytes BIGINT,
        mime_type TEXT NOT NULL DEFAULT 'application/pdf',
        approved_by UUID REFERENCES users(id) ON DELETE SET NULL,
        upload_status TEXT NOT NULL DEFAULT 'approved'
            CHECK (upload_status IN ('pending','approved','rejected')),
        processing_status TEXT NOT NULL DEFAULT 'pending'
            CHECK (processing_status IN ('pending','processing','completed','failed')),
        created_at TIMESTAMP NOT NULL DEFAULT NOW());""")
    # T4 documents
    cur.execute("""CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        title TEXT, file_name TEXT NOT NULL, file_path TEXT NOT NULL,
        department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
        uploaded_by UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
        source_user_upload_id UUID REFERENCES user_uploads(id) ON DELETE SET NULL,
        source_admin_upload_id UUID REFERENCES admin_uploads(id) ON DELETE SET NULL,
        embed_status TEXT NOT NULL DEFAULT 'pending'
            CHECK (embed_status IN ('pending','processing','completed','failed')),
        content_hash TEXT, page_count INTEGER NOT NULL DEFAULT 0,
        ocr_used BOOLEAN NOT NULL DEFAULT FALSE,
        version INTEGER NOT NULL DEFAULT 1,
        last_embedded_at TIMESTAMP);""")
    # T4 chunks
    cur.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        source_user_upload_id UUID REFERENCES user_uploads(id) ON DELETE SET NULL,
        source_admin_upload_id UUID REFERENCES admin_uploads(id) ON DELETE SET NULL,
        chunk_index INTEGER NOT NULL, chunk_text TEXT NOT NULL,
        chunk_token_count INTEGER, page_num INTEGER NOT NULL DEFAULT 0,
        doc_version INTEGER NOT NULL DEFAULT 1);""")
    # Migration
    for _sql in [
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_num INTEGER NOT NULL DEFAULT 0;",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_token_count INTEGER;",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_version INTEGER NOT NULL DEFAULT 1;",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_user_upload_id UUID;",
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_admin_upload_id UUID;",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INTEGER NOT NULL DEFAULT 0;",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_used BOOLEAN NOT NULL DEFAULT FALSE;",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;",
        "ALTER TABLE documents DROP COLUMN IF EXISTS is_shared_globally;",
        "ALTER TABLE admin_uploads DROP COLUMN IF EXISTS is_shared_globally;",
    ]:
        try: cur.execute(_sql)
        except Exception: pass
    # T5 embeddings
    cur.execute(f"""CREATE TABLE IF NOT EXISTS embeddings (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        source_user_upload_id UUID REFERENCES user_uploads(id) ON DELETE SET NULL,
        source_admin_upload_id UUID REFERENCES admin_uploads(id) ON DELETE SET NULL,
        department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
        embedding vector({EMBEDDING_DIM}),
        embedding_model TEXT NOT NULL DEFAULT '{EMBEDDING_MODEL}',
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        UNIQUE (chunk_id));""") # ◀ ADDED UNIQUE CONSTRAINT FOR UPSERTS
    # T5 rag_retrieval_log
    cur.execute("""CREATE TABLE IF NOT EXISTS rag_retrieval_log (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        chat_id UUID NOT NULL REFERENCES chat(id) ON DELETE CASCADE,
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        department_id UUID NOT NULL REFERENCES departments(id) ON DELETE CASCADE,
        query_text TEXT NOT NULL,
        retrieved_chunk_ids JSONB NOT NULL DEFAULT '[]',
        similarity_scores JSONB NOT NULL DEFAULT '[]',
        created_at TIMESTAMP NOT NULL DEFAULT NOW());""")
    # T6 admin_actions
    cur.execute("""CREATE TABLE IF NOT EXISTS admin_actions (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        admin_user_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
        department_id UUID NOT NULL REFERENCES departments(id) ON DELETE RESTRICT,
        action_type TEXT NOT NULL, target_type TEXT, target_id UUID,
        role_at_action TEXT, ip_address INET, metadata JSONB DEFAULT '{}',
        created_at TIMESTAMP NOT NULL DEFAULT NOW());""")

    # Indexes
    for sql in [
        "CREATE INDEX IF NOT EXISTS idx_dag_receiving  ON dept_access_grants(receiving_dept_id);",
        "CREATE INDEX IF NOT EXISTS idx_dag_granting   ON dept_access_grants(granting_dept_id);",
        "CREATE INDEX IF NOT EXISTS idx_users_dept     ON users(department_id);",
        "CREATE INDEX IF NOT EXISTS idx_chat_dept      ON chat(department_id);",
        "CREATE INDEX IF NOT EXISTS idx_msg_chat       ON messages(chat_id);",
        "CREATE INDEX IF NOT EXISTS idx_uu_dept        ON user_uploads(department_id);",
        "CREATE INDEX IF NOT EXISTS idx_au_dept        ON admin_uploads(department_id);",
        "CREATE INDEX IF NOT EXISTS idx_doc_dept       ON documents(department_id);",
        "CREATE INDEX IF NOT EXISTS idx_doc_hash       ON documents(content_hash);",
        "CREATE INDEX IF NOT EXISTS idx_chunk_doc      ON chunks(document_id);",
        "CREATE INDEX IF NOT EXISTS idx_emb_dept       ON embeddings(department_id);",
        "CREATE INDEX IF NOT EXISTS idx_rrl_chat       ON rag_retrieval_log(chat_id);",
        "CREATE INDEX IF NOT EXISTS idx_aa_dept        ON admin_actions(department_id);",
    ]:
        cur.execute(sql)
    cur.close()
    
    # Delegate HNSW index creation to our VectorIndexer
    indexer = VectorIndexer(conn)
    indexer.create_hnsw_index()
    
    logger.info("[Schema] 16 tables + indexes ready ✓")

class RBACManager:
    def __init__(self, conn):
        # Determine if we were passed a connection pool or a single connection
        if hasattr(conn, 'getconn') and hasattr(conn, 'putconn'):
            self.pool = conn
            self.conn = None
        else:
            self.pool = None
            self.conn = conn
        
        self.vectors = PGVectorStore(conn) # ◀ SPECIALIZED VECTOR ENGINE

    def _get_conn(self):
        """Retrieves a connection from the pool or returns the direct connection."""
        if self.pool:
            try:
                conn = self.pool.getconn()
                # Check if connection is still alive
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    logger.warning(f"Closing dead connection from pool: {e}")
                    self.pool.putconn(conn, close=True)
                    conn = self.pool.getconn()
                
                # Rollback any pending transaction before setting autocommit
                if conn.status != psycopg2.extensions.STATUS_READY:
                    conn.rollback()
                conn.autocommit = True
                return conn
            except Exception as e:
                logger.error(f"Failed to get connection from pool: {e}")
                raise
        if self.conn:
             self.conn.autocommit = True
             return self.conn
        return None

    def _put_conn(self, conn):
        if self.pool and conn:
            try:
                self.pool.putconn(conn)
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")

    def _cur(self, conn=None):
        """Provides a dictionary-like cursor with error handling."""
        try:
            c = conn or self._get_conn()
            return c.cursor(cursor_factory=RealDictCursor)
        except Exception as e:
            logger.error(f"Failed to create cursor: {e}")
            raise

    def _audit(self, admin_user_id, department_id, action_type,
               target_type=None, target_id=None, metadata=None):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""INSERT INTO admin_actions
                           (admin_user_id,department_id,action_type,target_type,target_id,metadata)
                           VALUES (%s,%s,%s,%s,%s,%s)""",
                        (admin_user_id, department_id, action_type,
                         target_type, target_id, Json(metadata or {})))
            cur.close()
        finally:
            self._put_conn(conn)

    def create_department(self, name, description=None, created_by=None):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("INSERT INTO departments (name,description,created_by) VALUES (%s,%s,%s) RETURNING id",
                        (name, description, created_by))
            r = str(cur.fetchone()["id"]); cur.close(); return r
        finally:
            self._put_conn(conn)

    def grant_dept_access(self, granting_dept_id, receiving_dept_id, granted_by,
                          access_type="read", expires_at=None):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""INSERT INTO dept_access_grants
                           (granting_dept_id,receiving_dept_id,granted_by,access_type,expires_at)
                           VALUES (%s,%s,%s,%s,%s)
                           ON CONFLICT (granting_dept_id,receiving_dept_id)
                           DO UPDATE SET access_type=EXCLUDED.access_type,granted_by=EXCLUDED.granted_by
                           RETURNING id""",
                        (granting_dept_id, receiving_dept_id, granted_by, access_type, expires_at))
            r = str(cur.fetchone()["id"]); cur.close()
            self._audit(granted_by, granting_dept_id, "dept_access_grant",
                        "department", receiving_dept_id, {"access_type": access_type})
            return r
        finally:
            self._put_conn(conn)

    def create_user(self, email, name, password_hash, department_id, is_super_admin=False):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""INSERT INTO users (email,name,password_hash,department_id,is_super_admin)
                           VALUES (%s,%s,%s,%s,%s) RETURNING id""",
                        (email, name, password_hash, department_id, is_super_admin))
            r = str(cur.fetchone()["id"]); cur.close(); return r
        finally:
            self._put_conn(conn)

    def create_chat(self, user_id, department_id, title=None, rag_enabled=True):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("INSERT INTO chat (user_id,department_id,title,rag_enabled) VALUES (%s,%s,%s,%s) RETURNING id",
                        (user_id, department_id, title, rag_enabled))
            r = str(cur.fetchone()["id"]); cur.close(); return r
        finally:
            self._put_conn(conn)

    def add_message(self, chat_id, role, content):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("INSERT INTO messages (chat_id,role,content) VALUES (%s,%s,%s) RETURNING id",
                        (chat_id, role, content))
            r = str(cur.fetchone()["id"]); cur.close(); return r
        finally:
            self._put_conn(conn)

    def register_user_upload(self, upload_id, user_id, dept_id, file_name, file_path, file_size_bytes, chat_id=None):
        logger.info(f"[DB] Registering User Upload: {upload_id} for user {user_id} in {dept_id}")
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""INSERT INTO user_uploads
                           (id, user_id, department_id, chat_id, file_name, file_path,
                            file_size_bytes, processing_status)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,'processing') RETURNING id""",
                        (upload_id, user_id, dept_id, chat_id, file_name, file_path,
                         file_size_bytes))
            r = str(cur.fetchone()["id"])
            cur.close()
            logger.info(f"[DB] SUCCESS: User Upload registered at {r}")
            self._audit(user_id, dept_id, "user_upload", "file",
                        metadata={"file_name": file_name})
            return r
        except Exception as e:
            logger.error(f"[DB] ERROR registering user upload: {e}")
            raise
        finally:
            self._put_conn(conn)

    def create_admin_upload(self, admin_user_id, dept_id, file_name, file_path,
                              file_size_bytes=None):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""INSERT INTO admin_uploads
                           (admin_user_id,department_id,file_name,file_path,
                            file_size_bytes,mime_type)
                           VALUES (%s,%s,%s,%s,%s,'application/pdf') RETURNING id""",
                        (admin_user_id, dept_id, file_name, file_path,
                         file_size_bytes))
            r = str(cur.fetchone()["id"]); cur.close()
            self._audit(admin_user_id, dept_id, "admin_upload", "file",
                        metadata={"file_name": file_name})
            return r
        finally:
            self._put_conn(conn)

    def update_upload_status(self, upload_id, upload_type, status):
        table = "user_uploads" if upload_type == "user" else "admin_uploads"
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute(f"UPDATE {table} SET processing_status=%s WHERE id=%s", (status, upload_id))
            cur.close()
        finally:
            self._put_conn(conn)

    def create_document(self, file_name, file_path, dept_id, uploaded_by,
                        content_hash=None, page_count=0, ocr_used=False,
                        source_user_upload_id=None, source_admin_upload_id=None):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""INSERT INTO documents
                           (title,file_name,file_path,department_id,uploaded_by,
                            content_hash,page_count,ocr_used,
                            source_user_upload_id,source_admin_upload_id)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (file_name, file_name, file_path, dept_id, uploaded_by,
                         content_hash, page_count, ocr_used,
                         source_user_upload_id, source_admin_upload_id))
            r = str(cur.fetchone()["id"]); cur.close(); return r
        finally:
            self._put_conn(conn)

    def update_document_status(self, doc_id, status):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""UPDATE documents SET embed_status=%s,
                           last_embedded_at=CASE WHEN %s='completed' THEN NOW()
                           ELSE last_embedded_at END WHERE id=%s""",
                        (status, status, doc_id))
            cur.close()
        finally:
            self._put_conn(conn)

    def add_chunk(self, doc_id, chunk_index, chunk_text, chunk_token_count,
                  page_num=0, source_user_upload_id=None, source_admin_upload_id=None):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""INSERT INTO chunks
                           (document_id,chunk_index,chunk_text,chunk_token_count,page_num,
                            source_user_upload_id,source_admin_upload_id)
                           VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (doc_id, chunk_index, chunk_text, chunk_token_count, page_num,
                         source_user_upload_id, source_admin_upload_id))
            r = str(cur.fetchone()["id"]); cur.close(); return r
        finally:
            self._put_conn(conn)

    def store_embedding(self, chunk_id, department_id, embedding, user_upload_id=None, admin_upload_id=None):
        return self.vectors.store_embedding(chunk_id, department_id, embedding, user_upload_id, admin_upload_id)

    def find_doc_by_hash(self, content_hash, dept_id):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""SELECT id FROM documents
                           WHERE content_hash=%s AND department_id=%s
                             AND embed_status='completed' LIMIT 1""",
                        (content_hash, dept_id))
            row = cur.fetchone(); cur.close()
            return str(row["id"]) if row else None
        finally:
            self._put_conn(conn)

    def vector_search(self, query_embedding, dept_id, top_k=20):
        return self.vectors.vector_search(query_embedding, dept_id, top_k)

    def log_retrieval(self, chat_id, user_id, dept_id, query_text, chunk_ids, scores):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""INSERT INTO rag_retrieval_log
                           (chat_id,user_id,department_id,query_text,
                            retrieved_chunk_ids,similarity_scores)
                           VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (chat_id, user_id, dept_id, query_text,
                         Json(chunk_ids), Json(scores)))
            r = str(cur.fetchone()["id"]); cur.close(); return r
        finally:
            self._put_conn(conn)

    def can_dept_see(self, viewing_dept_id, owning_dept_id, user_id=None):
        """Cross-department access check with optional auditing for deny events."""
        if viewing_dept_id == owning_dept_id: return True
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""SELECT 1 FROM dept_access_grants
                           WHERE granting_dept_id=%s AND receiving_dept_id=%s
                             AND (expires_at IS NULL OR expires_at>NOW()) LIMIT 1""",
                        (owning_dept_id, viewing_dept_id))
            r = cur.fetchone() is not None
            cur.close()
            
            # Compliance audit: Log access denied if a user_id is provided
            if not r and user_id:
                self._audit(user_id, viewing_dept_id, "access_denied", 
                            "department", owning_dept_id, 
                            {"viewing_dept": viewing_dept_id, "owning_dept": owning_dept_id})
            return r
        finally:
            self._put_conn(conn)

    def get_visible_depts(self, dept_id):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            cur.execute("""SELECT granting_dept_id::TEXT AS id FROM dept_access_grants
                           WHERE receiving_dept_id=%s AND (expires_at IS NULL OR expires_at>NOW())
                           UNION SELECT %s AS id""", (dept_id, dept_id))
            r = [row["id"] for row in cur.fetchall()]; cur.close(); return r
        finally:
            self._put_conn(conn)

    def get_audit_log(self, dept_id=None, limit=50):
        conn = self._get_conn()
        try:
            cur = self._cur(conn)
            conds, params = ["1=1"], []
            if dept_id: conds.append("department_id=%s"); params.append(dept_id)
            params.append(limit)
            cur.execute(f"SELECT * FROM admin_actions WHERE {' AND '.join(conds)} "
                        f"ORDER BY created_at DESC LIMIT %s", params)
            rows = []
            for row in cur.fetchall():
                item = dict(row)
                for k, v in item.items():
                    if not isinstance(v, (str, int, float, bool, type(None), dict, list)):
                        item[k] = str(v)
                rows.append(item)
            cur.close(); return rows
        finally:
            self._put_conn(conn)

def seed_rbac(rbac):
    """Seeds default metadata (departments and a test user) if the database is empty."""
    conn = rbac.conn
    with conn.cursor() as cur:
        # 1. Create Default Department
        cur.execute("SELECT id FROM departments WHERE name = 'Engineering' LIMIT 1")
        dept = cur.fetchone()
        if not dept:
            cur.execute("INSERT INTO departments (name, description) VALUES ('Engineering', 'Core engineering team') RETURNING id")
            dept_id = cur.fetchone()[0]
        else:
            dept_id = dept[0]

        # 2. Create Default User
        cur.execute("SELECT id FROM users WHERE email = 'admin@enterprise.ai' LIMIT 1")
        user = cur.fetchone()
        if not user:
            cur.execute(
                "INSERT INTO users (email, name, password_hash, department_id, is_super_admin) "
                "VALUES ('admin@enterprise.ai', 'System Admin', 'pbkdf2:sha256:260000', %s, TRUE) RETURNING id",
                (dept_id,)
            )
            user_id = cur.fetchone()[0]
        else:
            user_id = user[0]

        return {
            "dept_id": str(dept_id),
            "user_id": str(user_id)
        }
