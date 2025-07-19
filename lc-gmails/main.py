from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from auth import login, callback, logout, get_credentials
from email_processor import process_emails
from config import Config

app = FastAPI(title="Gmail Auto-Reply Service")
app.add_middleware(SessionMiddleware, secret_key=Config.SESSION_SECRET_KEY)

@app.get("/auth/login")
async def auth_login():
    return await login()

@app.get("/auth/callback")
async def auth_callback(request: Request):
    return await callback(request)

@app.get("/auth/logout")
async def auth_logout(request: Request):
    return await logout(request)

@app.get("/process-emails")
async def process_emails_endpoint(request: Request):
    creds = get_credentials(request)
    return await process_emails(creds)