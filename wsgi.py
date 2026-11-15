"""WSGI entry point: ``flask --app wsgi run`` or ``gunicorn wsgi:app``."""

from app import create_app
from app.logging_setup import configure_logging

configure_logging()
app = create_app()
