import os
from dotenv import load_dotenv
import pathlib
import uuid
import base64
# import torch
import asyncio
# from diffusers import StableDiffusionPipeline
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(
    model=os.environ["GEMINI_IMAGE_GENERATION"],
    google_api_key=os.environ["NAHOMY_GEMINI_KEY"],
)


async def generate_image(user_message: str) -> str:
    """
    Generates an image based on the user's message using a stable diffusion model.

    Args:
        user_message (str): The message input from the user.

    Returns:
        str: The file path of the generated image saved as 'generated_image.png'.
    """
    message = {
        "role": "user",
        "content": user_message,
    }

    response = await llm.ainvoke(
        [message],
        generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
    )

    image_base64 = get_image_base64(response=response)

    generated_images_path = pathlib.Path("generated_images")
    generated_images_path.mkdir(exist_ok=True)

    image_path = generated_images_path/f"{uuid.uuid4()}.png"

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