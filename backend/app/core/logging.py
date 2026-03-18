import structlog
import logging

def setup_logging() -> None:
    """
    Configure structlog for JSON output.

    LEARNING: structured logs output one JSON object per line.
    Instead of: "Ingested file in 2.3s"
    You get:    {"event":"ingestion_complete","file":"manual.pdf",
                 "chunks":283,"latency_ms":2341,"level":"info"}

    This means you can grep, filter, and aggregate logs in tools
    like Datadog, CloudWatch, or even just jq on the command line.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

logger = structlog.get_logger("docpilot")