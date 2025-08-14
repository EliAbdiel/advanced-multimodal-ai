import chainlit as cl
import PyPDF2
import docx

from .image_processing import process_img
from .audio_processing import process_audio
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.llm_setup import get_openrouter_llm

async def extract_text_from_pdf(file: cl.File) -> str:
    file_name = file.name
    await cl.Message(content=f"Processing the PDF file: **`{file_name}`**... Please hold on!").send()

    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    return pdf_text

async def extract_text_from_docx(file: cl.File) -> str:
    file_name = file.name
    await cl.Message(content=f"Processing the Word document: **`{file_name}`**... Please wait!").send()

    doc = docx.Document(file.path)
    doc_text = "\n".join([para.text for para in doc.paragraphs])

    return doc_text

async def handle_file_processing(user_message: cl.Message) -> str:
    """
    Processes different types of files (PDF, DOCX, images, audio) and extracts/analyzes their content.
    For text documents (PDF/DOCX), it extracts text and uses an LLM to answer questions about the content.
    For images and audio, it processes them according to user prompts.
    Returns a string containing the processed result or LLM response.
    """
    pdf_files = [file for file in user_message.elements if file.mime == "application/pdf"]
    docx_files = [file for file in user_message.elements if file.mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    image_files = [file for file in user_message.elements if file.mime.startswith("image/")]
    audio_files = [file for file in user_message.elements if file.mime.startswith("audio/")]

    user_msg = "Give me a short description"

    text=""
    if pdf_files:
        file = pdf_files[0]
        text = await extract_text_from_pdf(file=file)
    elif docx_files:
        file = docx_files[0]
        text = await extract_text_from_docx(file=file)
    elif image_files:
        if user_message.content:
            user_message = user_message.content
        else:
            user_message = user_msg
        file = image_files[0]
        text = await process_img(file=file, user_message=user_message)
        return text
    elif audio_files:
        if user_message.content:
            user_message = user_message.content
        else:
            user_message = user_msg
        file = audio_files[0]
        text = await process_audio(file=file, user_message=user_message)
        return text
    else:
        raise ValueError("Unsupported file type")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    
    # Get the LLM
    llm = await get_openrouter_llm()

    # Define prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Invoke chain
    prompt = await custom_rag_prompt.ainvoke({"question": user_message.content, "context": texts})
    answer = await llm.ainvoke(prompt)

    return answer.content