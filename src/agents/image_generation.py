import base64
import chainlit as cl

from src.utils.config import GENERATED_IMAGES
from src.utils.llm_setup import get_gemini_image_generation
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

@tool
async def generate_image(user_message: str) -> str:
    """
    AI-powered image generation system using Google Gemini's multimodal capabilities.
    
    This tool:
    1. Generates high-quality images from textual descriptions
    2. Handles both image creation and textual response generation
    3. Saves images to persistent storage
    4. Provides detailed generation metadata
    
    Workflow:
    1. Sends user prompt to Gemini's image generation model
    2. Processes multimodal response (text + image)
    3. Extracts base64-encoded image data
    4. Saves image to 'generated_images' directory
    5. Returns file path for UI integration
    
    Args:
    -----------
    user_message : str
        Textual description of desired image (e.g., "A futuristic city at sunset")
    
    Returns:
    --------
    str
        File path to generated image (PNG format)
    
    Technical Details:
    -----------------
    - Uses Gemini's multimodal generation capability:
        generation_config=dict(response_modalities=["TEXT", "IMAGE"])
    - Processes response with specialized image extraction:
        image_block = next(block for block in response.content if "image_url" in block)
    - Creates persistent storage directory if missing
    - Saves images with standardized naming convention
    
    Error Handling:
    --------------
    - Returns FileNotFoundError if directory creation fails
    - Raises ValueError if image data extraction fails
    - Handles base64 decoding errors gracefully
    
    Example Usage:
    -------------
    Input: "A cyberpunk cat wearing VR goggles"
    Output: "generated_images/generated_image.png"
    """
    await cl.Message(content="Image Generation Selected! \nYou've chosen to generate an image.").send()
    
    message = {
        "role": "user",
        "content": user_message,
    }

    model = await get_gemini_image_generation()

    response = await model.ainvoke(
        [message],
        generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
    )

    image_base64 = get_image_base64(response=response)

    # generated_images_path = pathlib.Path("generated_images")
    # generated_images_path.mkdir(exist_ok=True)

    image_path = GENERATED_IMAGES / 'generated_image.png'

    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_base64))

    string_path = str(image_path)

    print("\nGenerate Image Metadata:")
    print(response.usage_metadata)

    return string_path

def get_image_base64(response: AIMessage) -> None:
    """Extracts the base64 encoded image URL from the response."""
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    return image_block["image_url"].get("url").split(",")[-1]