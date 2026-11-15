"""Small operational commands: ``flask create-user`` and ``flask seed-demo``."""

from __future__ import annotations

import click
from flask import Flask, current_app

from app.config import DEFAULT_KB_SETTINGS
from app.db import db
from app.models import KnowledgeBase, User
from app.security import hash_password


def register_cli(app: Flask) -> None:
    @app.cli.command("create-user")
    @click.argument("email")
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
    @click.option("--name", default=None)
    def create_user(email: str, password: str, name: str | None) -> None:
        """Create a user and their starter knowledge base."""
        email = email.strip().lower()
        settings = current_app.config["SETTINGS"]
        if db.session.query(User).filter_by(email=email).first():
            raise click.ClickException(f"{email} already exists")
        user = User(email=email, password_hash=hash_password(password), display_name=name)
        db.session.add(user)
        db.session.flush()
        db.session.add(
            KnowledgeBase(
                user_id=user.id,
                name="My documents",
                embedding_model=settings.embedding_model,
                embedding_dim=settings.embedding_dim,
                **DEFAULT_KB_SETTINGS,
            )
        )
        db.session.commit()
        click.echo(f"created {email}")

    @app.cli.command("seed-demo")
    @click.option("--email", default="demo@example.com")
    @click.option("--password", default="demo-password")
    def seed_demo(email: str, password: str) -> None:
        """Idempotently create a demo account for local development."""
        email = email.strip().lower()
        if db.session.query(User).filter_by(email=email).first():
            click.echo(f"{email} already exists")
            return
        settings = current_app.config["SETTINGS"]
        user = User(email=email, password_hash=hash_password(password), display_name="Demo")
        db.session.add(user)
        db.session.flush()
        db.session.add(
            KnowledgeBase(
                user_id=user.id,
                name="Demo library",
                description="A sample knowledge base for local development.",
                embedding_model=settings.embedding_model,
                embedding_dim=settings.embedding_dim,
                **DEFAULT_KB_SETTINGS,
            )
        )
        db.session.commit()
        click.echo(f"created {email} / {password}")
