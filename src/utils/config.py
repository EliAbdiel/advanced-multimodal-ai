import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve the API key from environment variables
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_KEY_V2 = os.environ["GEMINI_API_KEY_V2"]
GEMINI_MODEL = os.environ["GEMINI_MODEL"]
GEMINI_VEO = os.environ["GEMINI_VEO"]
GEMINI_IMAGE_MODEL = os.environ["GEMINI_IMAGE_GENERATION"]
GEMINI_2_5_MODEL = os.environ["GEMINI_2_5_MODEL"]
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_KEY_V2 = os.environ["OPENROUTER_API_KEY_V2"]
OPENROUTER_MODEL = os.environ["OPENROUTER_MODEL_NAME"]
OPENROUTER_CODER = os.environ["OPENROUTER_CODER"]
OPENROUTER_URL = os.environ["OPENROUTER_BASE_URL"]

DATABASE_URL = os.environ["DATABASE_LOCAL_URL"]
MEMORY_DATABASE = os.environ["MEMORY_DATABASE"]

AZURE_CONTAINER_NAME = os.environ["CONTAINER_NAME"]
AZURE_STORAGE_ACCOUNT_NAME = os.environ["STORAGE_ACCOUNT_NAME"]
AZURE_STORAGE_KEY = os.environ["STORAGE_KEY"]

TAVILY_KEY = os.environ["TAVILY_API_KEY"]

ELEVENLABS_KEY = os.environ["ELEVENLABS_API_KEY"]

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

EXTRACTED_DATA = DATA_DIR / 'extracted_data'
GENERATED_IMAGES = DATA_DIR / 'generated_images'
GENERATED_VIDEOS = DATA_DIR / 'generated_videos'

# Usage in services:
# from utils.config import GENERATED_IMAGES
# image_path = GENERATED_IMAGES / "image.png"