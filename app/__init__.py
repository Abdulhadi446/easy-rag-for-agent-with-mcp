"""Flask app factory."""

from __future__ import annotations

from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask, g, render_template, session

from app.config import BASE_DIR, get_settings
from app.db import db
from app.models import EMBEDDING_DIM


def create_app(test_config: dict | None = None) -> Flask:
    load_dotenv(BASE_DIR / ".env", override=False)
    settings = get_settings()

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.update(
        SECRET_KEY=settings.secret_key,
        DATABASE_URL=settings.database_url,
        SETTINGS=settings,
        MAX_CONTENT_LENGTH=settings.max_upload_bytes,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=False,
        PERMANENT_SESSION_LIFETIME=timedelta(days=14),
        CSRF_ENABLED=True,
        TEMPLATES_AUTO_RELOAD=True,
    )
    if test_config:
        app.config.update(test_config)

    active_settings = app.config["SETTINGS"]
    if active_settings.embedding_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"EMBEDDING_DIM={active_settings.embedding_dim} does not match the vector column "
            f"built from EMBEDDING_DIM={EMBEDDING_DIM}. Re-embed or recreate the database."
        )

    app.config["DATABASE_URL"] = active_settings.database_url
    db.init_app(app, active_settings.database_url)

    from app.auth.routes import bp as auth_bp
    from app.chat.routes import bp as chat_bp
    from app.csrf import csrf_protect, csrf_token
    from app.documents.routes import bp as documents_bp
    from app.knowledge_bases.routes import bp as kb_bp
    from app.knowledge_bases.settings_routes import bp as kb_settings_bp
    from app.tenancy import load_current_user
    from app.tokens.routes import bp as tokens_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(kb_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(kb_settings_bp)
    app.register_blueprint(tokens_bp)
    app.register_blueprint(chat_bp)

    app.jinja_env.globals["csrf_token"] = csrf_token
    app.jinja_env.globals["settings"] = active_settings

    @app.before_request
    def _before() -> None:
        load_current_user()

    csrf_protect(app)

    @app.teardown_appcontext
    def _teardown(exception: BaseException | None) -> None:
        db.remove()

    @app.context_processor
    def _inject_user() -> dict:
        return {"current_user": g.get("user")}

    @app.get("/")
    def index():
        if g.get("user") is None:
            from flask import redirect, url_for

            return redirect(url_for("auth.login"))
        from flask import redirect, url_for

        return redirect(url_for("knowledge_bases.index"))

    @app.errorhandler(404)
    def not_found(error):
        return render_template("errors/404.html"), 404

    @app.errorhandler(413)
    def too_large(error):
        return render_template("errors/413.html", limit=active_settings.max_upload_mb), 413

    @app.errorhandler(500)
    def server_error(error):
        db.remove()
        return render_template("errors/500.html"), 500

    from app.cli import register_cli

    register_cli(app)

    # Sessions are marked permanent so a login survives a browser restart.
    @app.before_request
    def _permanent() -> None:
        if session.get("user_id"):
            session.permanent = True

    return app
