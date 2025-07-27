import os
from dotenv import load_dotenv
import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.azure_blob import AzureBlobStorageClient
from process_user_files import handle_attachment
from process_user_message import process_user_message
from process_user_audios import process_audio_chunk, audio_answer
from commands import command_list
from starter import select_starters
from resume_chat import resume_chat
from typing import Dict, Optional


load_dotenv(dotenv_path=".env", override=True)


@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  """Callback function for OAuth authentication."""
  return default_user


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session with proper error handling.
    """
    try:
        # Initialize session variables
        cl.user_session.set("chain", None)
        cl.user_session.set("audio_buffer", None)
        cl.user_session.set("audio_mime_type", None)

        commands = await command_list()

        await cl.context.emitter.set_commands(commands)
        
        # await cl.Message(content="Welcome! I'm your multimodal AI assistant. You can send me text, audio, images, PDFs, or Word documents!").send()
        
    except Exception as e:
        print(f"Error initializing chat: {e}")
        await cl.Message(content="Error initializing chat session. Please refresh and try again.").send()

@cl.set_starters
async def set_starters():
    """
    Sets up the initial conversation starters/suggestions that appear when a chat begins.
    These starters help guide users on how to interact with the assistant.
    
    Returns:
        list: A list of starter messages/suggestions from the select_starters() function
    """
    return await select_starters()


@cl.on_audio_start
async def on_audio_start():
    """Handler to manage mic button click event"""
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk) -> None:
    """
    Handles incoming audio chunks during user input.

    Receives audio chunks, stores the audio data in a buffer, and 
    updates the session with the buffer.

    Parameters:
    ----------
    audio_chunk : InputAudioChunk
        The audio chunk to process.
    """
    await process_audio_chunk(chunk=chunk)


@cl.on_audio_end
async def on_audio_end(elements: list = None) -> None:
    """
    Processes the voice message and analyzes user intent.

    Converts the audio to text using the selected chat profile. 
    Handles document analysis (file attachments) and determines 
    user intent for chatbot functionalities. Returns text and 
    voice responses depending on attached file types and user intents.

    Parameters:
    ----------
    elements : list
        A list of elements related to the audio message.
    """
    await audio_answer(elements=elements or [])


@cl.on_message
async def on_message(user_message: cl.Message) -> None:
    """
    Processes text messages, file attachments, and user intent.

    Handles text input, detects files in the user's message, 
    and processes them. It also interacts with the LLM chat profile 
    to respond based on the attached files and user intent for 
    chatbot functionalities.

    Parameters:
    ----------
    user_message : Message
        The incoming message with potential file attachments.
    """
    await handle_attachment(user_message=user_message)
    await process_user_message(user_message=user_message)


storage_client = AzureBlobStorageClient(
    container_name=os.environ["CONTAINER_NAME"],
    storage_account=os.environ["STORAGE_ACCOUNT_NAME"],
    storage_key=os.environ["STORAGE_KEY"]
)

@cl.data_layer
def get_data_layer():
    """Initializes the SQLAlchemy data layer for Chainlit."""
    # ALTER TABLE steps ADD COLUMN "defaultOpen" BOOLEAN DEFAULT false;
    # ALTER TABLE elements ADD COLUMN "autoPlay" BOOLEAN DEFAULT false;
    return SQLAlchemyDataLayer(
        conninfo=os.environ["DATABASE_LOCAL_URL"], 
        # ssl_require=True,
        storage_provider=storage_client,
    )


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """
    Resumes archived chat conversations.

    Retrieves previous chat threads to load them into memory and 
    enables users to continue a conversation.

    Parameters:
    ----------
    thread : ThreadDict
        A dictionary containing the thread's information and messages.
    """
    await resume_chat(thread=thread)
    