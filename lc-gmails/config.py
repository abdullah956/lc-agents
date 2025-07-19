import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
class Config:
    OAUTHLIB_INSECURE_TRANSPORT = "1"  # Allow HTTP for development
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    REDIRECT_URI = os.getenv("REDIRECT_URI")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
    SESSION_SECRET_KEY = os.getenv("CLIENT_SECRET")

# Set environment variable for OAuth
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = Config.OAUTHLIB_INSECURE_TRANSPORT