import os
from dotenv import load_dotenv

def load_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
    required_vars = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID",
        "PGHOST",
        "PGPORT",
        "PGUSER",
        "PGPASSWORD",
        "PGDATABASE"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    return {
        "POSTGRES_URL": f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}",
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID")
    }