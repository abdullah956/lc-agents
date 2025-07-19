import os
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from base64 import urlsafe_b64decode, urlsafe_b64encode
from langchain_openai import ChatOpenAI

# Load env variables
load_dotenv()
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Allow HTTP for dev

app = FastAPI(title="Gmail Auto-Reply Service")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("CLIENT_SECRET"))

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
REDIRECT_URI = os.getenv("REDIRECT_URI")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=OPENAI_API_KEY)

# Step 1: OAuth Login
@app.get("/auth/login")
async def login():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uris": [REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline", include_granted_scopes="true")
    return RedirectResponse(auth_url)

# Step 2: OAuth Callback
@app.get("/auth/callback")
async def callback(request: Request):
    code = request.query_params.get("code")
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uris": [REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
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
    # Redirect to FastAPI docs after successful authentication
    return RedirectResponse(url="http://localhost:8000/docs")

# Step 3: Logout
@app.get("/auth/logout")
async def logout(request: Request):
    # Clear the session
    request.session.clear()
    # Redirect to docs page
    return RedirectResponse(url="http://localhost:8000/docs")

# Step 4: Process unread emails
@app.get("/process-emails")
async def process_emails(request: Request):
    creds_data = request.session.get("creds")
    if not creds_data:
        return {"error": "User not authenticated."}
    
    creds = Credentials(
        token=creds_data["token"],
        refresh_token=creds_data["refresh_token"],
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=creds_data["scopes"]
    )

    service = build("gmail", "v1", credentials=creds)
    
    # Get logged-in user's profile information
    profile = service.users().getProfile(userId="me").execute()
    user_email = profile.get("emailAddress", "")
    # Extract user's name from email (before @ symbol) - you could also store this during auth
    user_name = user_email.split("@")[0].replace(".", " ").title() if user_email else "Assistant"
    
    results = service.users().messages().list(userId="me", q="is:unread").execute()
    messages = results.get("messages", [])
    
    count = 0
    for msg in messages:
        msg_detail = service.users().messages().get(userId="me", id=msg["id"]).execute()
        thread_id = msg_detail["threadId"]
        payload = msg_detail.get("payload", {})
        headers = payload.get("headers", [])
        
        # Extract sender email and subject from headers
        sender_email = None
        original_subject = None
        for header in headers:
            if header["name"] == "From":
                sender_info = header["value"]
                # Extract email from "Name <email@domain.com>" format
                if "<" in sender_info and ">" in sender_info:
                    sender_email = sender_info.split("<")[1].split(">")[0].strip()
                else:
                    sender_email = sender_info
            elif header["name"] == "Subject":
                original_subject = header["value"]
        
        if not sender_email:
            continue  # Skip if no sender email found
        
        # Extract email body
        parts = payload.get("parts", [payload])
        body = ""

        for part in parts:
            data = part.get("body", {}).get("data")
            if data and part.get("mimeType") == "text/plain":
                body = urlsafe_b64decode(data).decode("utf-8")
                break

        if body:
            # Generate reply using LLM
            reply_text = llm.invoke(f"Generate a professional reply to this email:\n\n{body}").content
            
            # Replace placeholders with actual names
            reply_text = reply_text.replace("[Your Name]", user_name)
            
            message = MIMEText(reply_text)
            message["to"] = sender_email
            message["subject"] = f"Re: {original_subject}" if original_subject else "Re: Auto Reply"
            raw = urlsafe_b64encode(message.as_bytes()).decode()
            service.users().messages().send(userId="me", body={"raw": raw, "threadId": thread_id}).execute()
            count += 1

    return {"message": f"Processed {count} unread emails."}