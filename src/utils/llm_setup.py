from .config import (
    GEMINI_KEY,
    GEMINI_KEY_V2,
    GEMINI_MODEL,
    GEMINI_2_5_MODEL,
    GEMINI_IMAGE_MODEL,
)
from .config import (
    OPENROUTER_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_URL,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

async def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Initializes the Gemini LLM with the API key and model settings.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        rate_limiter=rate_limiter,
        google_api_key=GEMINI_KEY,
    )
    # print(f"\nGemini LLM type: {type(llm)}\n")
    return llm

async def get_gemini_llm_v2() -> ChatGoogleGenerativeAI:
    """
    Initializes the Gemini LLM V2 with the API key and model settings.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        rate_limiter=rate_limiter,
        google_api_key=GEMINI_KEY_V2,
    )
    # print(f"\nGemini LLM V2 type: {type(llm)}\n")
    return llm

async def get_openrouter_llm() -> ChatOpenAI:
    """
    Initializes and returns an OpenRouter LLM instance with configured settings.
    """
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        rate_limiter=rate_limiter,
        api_key=OPENROUTER_KEY,
        base_url=OPENROUTER_URL,
    )
    # print(f"\nOpenrouter LLM type: {type(llm)}\n")
    return llm

async def get_gemini_image_generation() -> ChatGoogleGenerativeAI:
    """
    Initializes the Gemini image generation model with the API key and model settings.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_IMAGE_MODEL,
        rate_limiter=rate_limiter,
        google_api_key=GEMINI_KEY,
    )
    # print(f"\nGemini image generation type: {type(llm)}\n")
    return llm

async def get_gemini_url_context(contents: str):
    """"Scrapes the content of a URL using Google Gemini's UrlContext tool."""
    client = await get_gemini_client()

    tools = []
    tools.append(Tool(url_context=types.UrlContext))
    tools.append(Tool(google_search=types.GoogleSearch()))

    gemini_client = client.models.generate_content(
        model=GEMINI_2_5_MODEL,
        contents=contents,
        config=GenerateContentConfig(
            tools=tools,
            response_modalities=["TEXT"],
        )
    )
    # print(f"\nGemini url with context type: {type(gemini_client)}\n")
    return gemini_client

async def get_gemini_llm_for_youtube(url: str, context: str):
    """
    Initializes the Gemini LLM for YouTube with the API key and model settings.
    """
    client = await get_gemini_client()

    llm = client.models.generate_content(
        model=GEMINI_2_5_MODEL,
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=url)
                ),
                types.Part(text=context)
            ]
        )
    )
    # print(f"\nGemini LLM for YouTube type: {type(llm)}\n")
    return llm

async def get_gemini_client():
    """Initializes and returns a Google Generative AI client with API key."""
    client = genai.Client(api_key=GEMINI_KEY)
    return client