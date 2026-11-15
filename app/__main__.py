"""Run the web app (and later the MCP server) from one command.

    python -m app            # Flask on :8000, MCP on :8001 once milestone 5 lands
    python -m app --mcp-only # only the remote MCP server
"""

from __future__ import annotations

import argparse
import threading

from app import create_app
from app.logging_setup import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG MCP platform")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mcp-only", action="store_true")
    parser.add_argument("--no-mcp", action="store_true")
    parser.add_argument("--no-worker", action="store_true", help="do not start ingestion threads")
    args = parser.parse_args()

    configure_logging()
    app = create_app()

    if not args.no_worker and app.config["SETTINGS"].worker_enabled:
        from ingestion.worker import IngestWorker

        worker = IngestWorker(app, concurrency=app.config["SETTINGS"].worker_concurrency)
        worker.start()
        print(f"Ingestion worker running ({app.config['SETTINGS'].worker_concurrency} threads)")

    mcp_started = False
    if not args.no_mcp:
        try:
            from mcp_server.server import run_mcp_server
        except ImportError:
            run_mcp_server = None
        if run_mcp_server is not None:
            threading.Thread(target=run_mcp_server, daemon=True, name="mcp-server").start()
            mcp_started = True
            print(f"MCP server listening on http://{app.config['SETTINGS'].mcp_host}:"
                  f"{app.config['SETTINGS'].mcp_port}/mcp")

    if args.mcp_only and mcp_started:
        threading.Event().wait()
        return

    app.run(host=args.host, port=args.port, debug=app.config["SETTINGS"].debug, use_reloader=False)


if __name__ == "__main__":
    main()
