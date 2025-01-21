# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")


# New environment variable for sandbox directory
SANDBOX_DIR = os.getenv("SANDBOX_DIR", "sandbox")  # Default to 'sandbox' if not set