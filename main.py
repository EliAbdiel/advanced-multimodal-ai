import chainlit as cl

from mcp import ClientSession
from typing import Dict, Optional
from chainlit.types import ThreadDict
from src.ui.starters import select_starter
from src.ui.commands import command_list
from src.utils.persistent_data_layer import get_persistent_data_layer
from src.services.speech_processing import run_audio_chunk, run_audio_workflow
from src.workflow import run_agent_workflow

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
        cl.user_session.set("audio_buffer", None)
        cl.user_session.set("audio_mime_type", None)

        commands = await command_list()

        await cl.context.emitter.set_commands(commands)
        
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
    return await select_starter()

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
    await run_audio_chunk(chunk=chunk)

@cl.on_audio_end
async def on_audio_end() -> None:
    """
    Processes the voice message and analyzes user intent.

    Converts the audio to text using the selected chat profile. 
    Handles document analysis (file attachments) and determines 
    user intent for chatbot functionalities. Returns text and 
    voice responses depending on attached file types and user intents.
    """
    await run_audio_workflow()

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    # Your connection initialization code here
    # This handler is required for MCP to work
    # TODO: implement connection logic
    pass
    
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    # Your cleanup code here
    # This handler is optional
    # TODO: implement disconnection logic
    pass

@cl.on_message
async def on_message(user_message: cl.Message) -> None:
    """
    Processes text messages, file attachments, and user intent.

    Handles text input, detects files in the user's message, 
    and processes them. It also interacts with the LLM chat profile 
    to respond based on the attached files and user intent for 
    chatbot functionalities.

    Args:
    ----------
    user_message : Message
        The incoming message with potential file attachments.
    """
    await run_agent_workflow(user_message=user_message)

@cl.data_layer
def get_data_layer():
    """Initializes the SQLAlchemy data layer for Chainlit."""
    return get_persistent_data_layer()

@cl.on_chat_resume
async def on_chat_resumen(thread: ThreadDict) -> None:
    """
    Resumes archived chat conversations.

    Retrieves previous chat threads to load them into memory and 
    enables users to continue a conversation.

    Args:
    ----------
    thread : ThreadDict
        A dictionary containing the thread's information and messages.
    """
    pass