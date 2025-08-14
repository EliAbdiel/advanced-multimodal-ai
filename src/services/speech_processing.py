import io
import wave
import audioop
import numpy as np
import chainlit as cl
import speech_recognition as sr

from typing import Dict, List, Optional, Union, cast
from elevenlabs.client import AsyncElevenLabs
from src.utils.config import ELEVENLABS_KEY
from src.workflow import run_agent_workflow

# Define a threshold for detecting silence and a timeout for ending a turn
SILENCE_THRESHOLD = (
    2000
    # 3500  # Adjust based on your audio level (e.g., lower for quieter audio)
)
SILENCE_TIMEOUT = 1300.0  # Seconds (1.3) of silence to consider the turn finished

class MessageObject:
    """Represents a message object with a content attribute."""
    __slots__ = ['content', 'command', 'elements']
    def __init__(
        self, 
        content: str = "", 
        command: Optional[str] = None, 
        elements: Optional[List] = None
    ):
        self.content = content
        self.command = command
        self.elements = elements

async def run_audio_chunk(chunk: cl.InputAudioChunk) -> None:
    """
    Handles incoming audio chunks and stores them in a buffer for further processing.

    Args:
        chunk (cl.InputAudioChunk): The audio data to process.

    Returns:
        BytesIO: The buffer containing the audio data.
    """
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    audio_chunks = cl.user_session.get("audio_chunks")
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(
        chunk.data, 2
    )  # Assumes 16-bit audio (2 bytes per sample)

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)

async def process_audio() -> None:
    """Enhanced audio processing with noise reduction and optimization"""

    # Get the audio buffer from the session
    if audio_chunks := cl.user_session.get("audio_chunks"):
        # Concatenate all chunks
        concatenated = np.concatenate(list(audio_chunks))

        # Create an in-memory binary stream
        wav_buffer = io.BytesIO()

        # Create WAV file with proper parameters
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # sample rate (24kHz PCM)
            wav_file.writeframes(concatenated.tobytes())

        # Reset buffer position
        wav_buffer.seek(0)

        cl.user_session.set("audio_chunks", [])

    frames = wav_file.getnframes()
    rate = wav_file.getframerate()

    duration = frames / float(rate)

    if duration <= 1.71:
        print("The audio is too short, please try again.")
        return

    wav_buffer.seek(0)
    cl.user_session.set("audio_buffer", wav_buffer)
    cl.user_session.set("audio_mime_type", "audio/wav")

async def speech_to_text(audio_file: bytes) -> str:
    """Enhanced transcription with multiple engines and languages"""
    # response = await openai_client.audio.transcriptions.create(
    #     model="whisper-1", file=audio_file
    # )
    # recognizer = sr.Recognizer()
    # with sr.AudioFile(audio_file) as source:
    #     # Enhanced ambient noise adjustment
    #     recognizer.adjust_for_ambient_noise(source, duration=0.2)
    #     audio = recognizer.record(source)
    #     response = recognizer.recognize_google(audio, language="es-ES", show_all=False)
    elevenlabs = AsyncElevenLabs(
        api_key=ELEVENLABS_KEY,
    )
    response = await elevenlabs.speech_to_text.convert(
        file=audio_file,
        model_id="scribe_v1", # Model to use, for now only "scribe_v1" is supported
        tag_audio_events=True, # Tag audio events like laughter, applause, etc.
        language_code="spa", # Language of the audio file. If set to None, the model will detect the language automatically.
        diarize=True, # Whether to annotate who is speaking
    )

    if response:
        print(f"\nTranscription response: {response}\n")

    return response.text

async def run_audio_workflow() -> None:
    """
    Processes the audio answer and sends a message with the transcription.
    """ 
    audio_buffer = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    
    if not audio_buffer:
        await cl.Message(content="Could not retrieve audio for processing. Please try recording again.").send()
        return
    
    try:
        # Enhanced recognizer settings
        transcription = await speech_to_text(audio_file=audio_buffer)
    
        if not transcription:
            await cl.Message(content="Could not understand the audio. Please try speaking more clearly or check your microphone.").send()
            return
        
        # Process transcription
        msg = MessageObject(transcription)
        print(f"\nMessage object: {msg.content}\n")
        await run_agent_workflow(msg) 

    except sr.UnknownValueError:
        await cl.Message(content="Unable to recognize speech. Please try again with clearer pronunciation.").send()
    except Exception as e:
        await cl.Message(content=f"Audio processing error: {str(e)}").send()