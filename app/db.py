"""SQLAlchemy engine/session plumbing shared by the web app, worker and MCP server."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from flask import Flask, current_app
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from app.models import Base


class Database:
    """A lazily-built engine plus one scoped session per thread/request."""

    def __init__(self) -> None:
        self._engine = None
        self._session_factory: sessionmaker | None = None
        self._scoped: scoped_session | None = None

    # ------------------------------------------------------------------ setup
    def init_app(self, app: Flask, url: str | None = None) -> None:
        database_url = url or app.config["DATABASE_URL"]
        self._engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            future=True,
        )
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False, autoflush=False, future=True
        )
        self._scoped = scoped_session(self._session_factory)
        app.extensions["db"] = self

    @property
    def engine(self):
        if self._engine is None:
            if current_app:
                return current_app.extensions["db"]._engine
            raise RuntimeError("Database is not initialised")
        return self._engine

    @property
    def session(self) -> Session:
        if self._scoped is None:
            if current_app:
                return current_app.extensions["db"]._scoped()
            raise RuntimeError("Database is not initialised")
        return self._scoped()

    def remove(self) -> None:
        if self._scoped is not None:
            self._scoped.remove()

    # -------------------------------------------------------------- utilities
    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """A session for background work: commit on success, rollback on error."""
        if self._session_factory is None:
            raise RuntimeError("Database is not initialised")
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.remove()
        if self._engine is not None:
            self._engine.dispose()


db = Database()


def get_db() -> Database:
    return db
