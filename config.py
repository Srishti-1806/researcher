# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    # --- Model Configuration: Groq API ---
    GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/v1/models/openai/v1/chat/completions")
    # Backwards-compatible alias used elsewhere in the code
    MODEL_NAME = GROQ_MODEL_NAME

    # Model parameters
    TEMPERATURE = 0.7
    MAX_NEW_TOKENS = 1024
    
    # --- API Keys ---
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # --- Phase 1: Budget & Constraints ---
    MAX_TOKENS_PER_QUERY = 5000 
    MAX_ITERATIONS_DEEP_MODE = 3
    CONFIDENCE_THRESHOLD = 0.8
    
    # --- Path Configuration ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    @staticmethod
    def validate():
        """Ensure critical config is present."""
        print(f"🔍 Configuration Check:")
        print(f"  ✓ Model: {Config.GROQ_MODEL_NAME}")
        print(f"  ✓ Groq API Key: {'Set' if Config.GROQ_API_KEY else 'Not Set'}")
        
        if not Config.TAVILY_API_KEY:
            print(f"  ⚠️  Tavily API Key not set - using DuckDuckGo for search")
        else:
            print(f"  ✓ Tavily API configured")
            
# Create directories if they don't exist
os.makedirs(Config.QDRANT_PATH, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)