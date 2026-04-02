import contextlib
import logging
import time
import uuid
from typing import Dict, Any, Optional

logger = logging.getLogger("tracing")

class Tracer:
    """
    Simulated Opentelemetry / Jaeger tracing utility.
    Manages active span propagation and operation IDs across the pipeline.
    """
    def __init__(self):
        # We track traces across the lifecycle of a single request
        self.active_trace_id: Optional[str] = None
        logger.info("Tracer initialized.")

    @contextlib.contextmanager
    def span(self, name: str, metadata: Dict[str, Any] = None):
        """
        Context manager for defining a traceable span.
        Each span must be tied to an active trace_id.
        """
        if not self.active_trace_id:
            self.active_trace_id = str(uuid.uuid4())
            logger.info(f"Root span created: {name} (Trace-ID: {self.active_trace_id})")
        
        start_time = time.time()
        logger.debug(f"Span started: {name} (Metadata: {metadata or {}})")
        
        try:
            yield self.active_trace_id
        except Exception as e:
            logger.error(f"Span ERROR in {name}: {e} (Trace-ID: {self.active_trace_id})")
            raise e
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Span finished: {name} | Duration: {duration:.4f}s | Trace-ID: {self.active_trace_id}")

    def reset(self):
        """Used to clear session-level tracing after a batch finishes."""
        self.active_trace_id = None

tracer = Tracer()

def instrument_app(app: Any):
    """
    Middleware stub to hook into FastAPI for global tracing on every request.
    """
    # Simply log instrumentation status for now
    logger.info("FastAPI application instrumented for OpenTelemetry-ready tracing.")
