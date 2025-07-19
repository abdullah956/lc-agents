from googleapiclient.discovery import build
from email.mime.text import MIMEText
from base64 import urlsafe_b64decode, urlsafe_b64encode
from langchain_openai import ChatOpenAI
from config import Config

def initialize_llm():
    """Initialize the language model."""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=Config.OPENAI_API_KEY)

async def process_emails(creds):
    """Process unread emails and send auto-replies."""
    if not creds:
        return {"error": "User not authenticated."}

    service = build("gmail", "v1", credentials=creds)
    llm = initialize_llm()

    # Get user profile
    profile = service.users().getProfile(userId="me").execute()
    user_email = profile.get("emailAddress", "")
    user_name = user_email.split("@")[0].replace(".", " ").title() if user_email else "Assistant"

    # Get unread messages
    results = service.users().messages().list(userId="me", q="is:unread").execute()
    messages = results.get("messages", [])
    
    count = 0
    for msg in messages:
        msg_detail = service.users().messages().get(userId="me", id=msg["id"]).execute()
        thread_id = msg_detail["threadId"]
        payload = msg_detail.get("payload", {})
        headers = payload.get("headers", [])
        
        # Extract sender email and subject
        sender_email = None
        original_subject = None
        for header in headers:
            if header["name"] == "From":
                sender_info = header["value"]
                if "<" in sender_info and ">" in sender_info:
                    sender_email = sender_info.split("<")[1].split(">")[0].strip()
                else:
                    sender_email = sender_info
            elif header["name"] == "Subject":
                original_subject = header["value"]
        
        if not sender_email:
            continue
        
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
            reply_text = reply_text.replace("[Your Name]", user_name)
            
            # Send reply
            message = MIMEText(reply_text)
            message["to"] = sender_email
            message["subject"] = f"Re: {original_subject}" if original_subject else "Re: Auto Reply"
            raw = urlsafe_b64encode(message.as_bytes()).decode()
            service.users().messages().send(userId="me", body={"raw": raw, "threadId": thread_id}).execute()
            count += 1

    return {"message": f"Processed {count} unread emails."}