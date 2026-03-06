import os
import requests
import time
from config import Config
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
def call_groq_api(url, json_payload, headers):
    """Make HTTP request to Groq API with automatic retry on rate limit (429) and transient errors."""
    resp = requests.post(url, json=json_payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp


class GroqClient:
    """Minimal Groq API client wrapper.

    This is a thin HTTP wrapper that posts inference requests to a configurable
    Groq-style REST endpoint. Adjust `GROQ_API_URL` in the environment if your
    provider uses a different base path.
    """

    def __init__(self, model: str = None, api_key: str = None, api_url: str = None, max_new_tokens: int = 512, temperature: float = 0.7):
        self.model = model or Config.GROQ_MODEL_NAME
        self.api_key = api_key or Config.GROQ_API_KEY
        self.api_url = api_url or Config.GROQ_API_URL
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set in environment")

    def generate(self, prompt: str, max_new_tokens: int = None, temperature: float = None) -> str:
        """Send an inference request and return text output (OpenAI-chat compatible).

        This method supports both a simple string `prompt` (converted into a single
        user message) or a `messages` list matching the OpenAI/chat schema.
        It POSTS to the Groq OpenAI-compatible chat completions endpoint by
        default: `{GROQ_API_URL}/openai/v1/chat/completions`.
        """

        # Normalize messages
        if isinstance(prompt, list):
            messages = prompt
        elif isinstance(prompt, dict) and "messages" in prompt:
            messages = prompt["messages"]
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        # Build endpoint URL (accept either a base URL or a full OpenAI-like path)
        base = self.api_url.rstrip('/')
        if base.endswith('/chat/completions') or base.endswith('/openai/v1/chat/completions'):
            url = base
        elif 'openai' in base:
            # If user provided an OpenAI-like base, append the completions path
            url = f"{base}/chat/completions"
        else:
            # Default Groq OpenAI-compatible path
            url = f"{base}/openai/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
        }
        if max_new_tokens:
            payload["max_tokens"] = max_new_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = call_groq_api(url, payload, headers)
            data = resp.json()

            # OpenAI-like response: choices -> message -> content
            if isinstance(data, dict) and "choices" in data:
                texts = []
                for c in data.get("choices", []):
                    msg = c.get("message") or {}
                    text = msg.get("content") or c.get("text")
                    if text:
                        texts.append(text)
                return "\n".join(texts).strip()

            # Fallbacks for other response shapes
            for key in ("output", "generated_text", "text", "result"):
                if isinstance(data, dict) and key in data:
                    return data[key]
            if isinstance(data, list):
                return str(data)
            return str(data)
        except RetryError as e:
            print(f"⚠️ Groq API unavailable after 5 retries (rate limited or service issue)")
            return "[Fallback] Groq API is rate limited. Please try your query again in a moment."
        except requests.exceptions.Timeout:
            return "[Timeout] The Groq API request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "[Connection Error] Unable to connect to Groq API. Please check your network."
        except Exception as e:
            print(f"⚠️ Error calling Groq API: {type(e).__name__}: {e}")
            return f"[API Error] {str(e)[:100]}"

    def invoke(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, max_new_tokens=kwargs.get("max_new_tokens"), temperature=kwargs.get("temperature"))


class DummyLLM:
    """A very small fallback LLM used when no Groq API key is provided.

    It returns predictable, short answers so the system can continue running
    for development and debugging without external API access.
    """
    def __init__(self, model_name: str = "dummy"):
        self.model = model_name

    def generate(self, prompt: str, max_new_tokens: int = None, temperature: float = None) -> str:
        # Keep outputs brief and deterministic for testing
        if "classify" in prompt.lower() or "classify the following query" in prompt.lower():
            return "Category: Research, Clear: True"
        if "deep" in prompt.lower() and "quick" in prompt.lower():
            return "quick"
        # Generic echo-style response
        return "[DUMMY LLM] This is a development fallback response."

    def invoke(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, max_new_tokens=kwargs.get("max_new_tokens"), temperature=kwargs.get("temperature"))


def get_llm():
    """Return a live GroqClient if API key present, otherwise a DummyLLM."""
    from config import Config
    try:
        if Config.GROQ_API_KEY:
            return GroqClient(Config.GROQ_MODEL_NAME, api_key=Config.GROQ_API_KEY, api_url=Config.GROQ_API_URL, max_new_tokens=Config.MAX_NEW_TOKENS, temperature=Config.TEMPERATURE)
    except Exception:
        pass
    return DummyLLM()
