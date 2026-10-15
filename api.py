"""Entry point:  uvicorn api:app --reload   (same command as the class project)"""
from app.main import build_default_app

app = build_default_app()
