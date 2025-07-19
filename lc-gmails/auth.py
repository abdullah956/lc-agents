from fastapi import Request
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from config import Config

def create_flow():
    """Create OAuth flow for Gmail API."""
    return Flow.from_client_config(
        {
            "web": {
                "client_id": Config.CLIENT_ID,
                "client_secret": Config.CLIENT_SECRET,
                "redirect_uris": [Config.REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=Config.SCOPES,
        redirect_uri=Config.REDIRECT_URI
    )

async def login():
    """Initiate OAuth login."""
    flow = create_flow()
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline", include_granted_scopes="true")
    return RedirectResponse(auth_url)

async def callback(request: Request):
    """Handle OAuth callback and store credentials in session."""
    code = request.query_params.get("code")
    flow = create_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    request.session["creds"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes
    }
    return RedirectResponse(url="http://localhost:8000/docs")

async def logout(request: Request):
    """Clear session and log out."""
    request.session.clear()
    return RedirectResponse(url="http://localhost:8000/docs")

def get_credentials(request: Request):
    """Retrieve credentials from session."""
    creds_data = request.session.get("creds")
    if not creds_data:
        return None
    return Credentials(
        token=creds_data["token"],
        refresh_token=creds_data["refresh_token"],
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=creds_data["scopes"]
    )