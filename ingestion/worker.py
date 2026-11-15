"""Small durable worker: claims ingestion jobs from Postgres and runs them.

Threads in the web process are enough for the target scale, and no Redis/Celery
deployment is required. ``python -m ingestion.worker`` runs a worker-only process.
"""

from __future__ import annotations

import logging
import threading
import time

from flask import Flask

from app.db import db
from ingestion.pipeline import process_next_job, reset_stuck_jobs

log = logging.getLogger(__name__)


class IngestWorker:
    def __init__(self, app: Flask, concurrency: int = 2, poll_interval: float = 0.75) -> None:
        self.app = app
        self.concurrency = max(1, concurrency)
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        with self.app.app_context():
            try:
                reset_stuck_jobs()
            except Exception:  # noqa: BLE001
                log.exception("could not reset stuck jobs")
        for index in range(self.concurrency):
            thread = threading.Thread(
                target=self._loop, name=f"ingest-worker-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        log.info("ingestion worker started with %s threads", self.concurrency)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                with self.app.app_context():
                    worked = process_next_job()
                    db.remove()
            except Exception:  # noqa: BLE001
                log.exception("worker loop error")
                worked = False
            if not worked:
                self._stop.wait(self.poll_interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()
        self._started = False


def run_worker_forever(app: Flask) -> None:
    concurrency = int(app.config["SETTINGS"].worker_concurrency)
    worker = IngestWorker(app, concurrency=concurrency, poll_interval=1.0)
    worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()


def main() -> None:
    from app import create_app
    from app.logging_setup import configure_logging

    configure_logging()
    app = create_app()
    with app.app_context():
        reset_stuck_jobs()
    run_worker_forever(app)


if __name__ == "__main__":
    main()
