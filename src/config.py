import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemini-1.5-flash")

# OpenRouter endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Experiment Configuration
NUM_TRIALS = 5
TEMPERATURE = 0.7

# File Paths
PROMPTS_DIR = "prompts"
QUERIES_FILE = "queries/test_queries.txt"
OUTPUT_DIR = "outputs/responses"