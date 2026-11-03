"""Centralised model config — single place to update Claude model IDs."""
from dotenv import load_dotenv
load_dotenv()

import os
MODEL_FAST      = os.getenv("MODEL_FAST",      "claude-haiku-4-5-20251001")
MODEL_SYNTHESIS = os.getenv("MODEL_SYNTHESIS",  "claude-sonnet-4-6")
