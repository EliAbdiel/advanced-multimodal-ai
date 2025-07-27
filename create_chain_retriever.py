import os
import getpass
import chainlit as cl
from dotenv import load_dotenv
# from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ConversationalRetrievalChain


load_dotenv(override=True)


llm = ChatGoogleGenerativeAI(
    model=os.environ["GEMINI_MODEL"],
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.5,
)

# llm = ChatOllama(model="llama3.2:1b")

async def create_chain_retriever(texts: str, source_prefix: str) -> ConversationalRetrievalChain:
    """
    Creates a conversational retrieval chain for processing texts.

    Splits the input texts, indexes them with embeddings, and prepares a conversational retrieval chain 
    that can retrieve relevant information while maintaining conversation context.

    Args:
        texts (str): The input text or collection of texts to process.
        source_prefix (str): A prefix for generating metadata for each text chunk.

    Returns:
        ConversationalRetrievalChain: A configured retrieval chain for text-based interactions.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(texts)
    metadatas = [{"source": f"{source_prefix}-{i}"} for i in range(len(texts))]
    embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=os.environ["GEMINI_API_KEY"],
                )
    docsearch = await cl.make_async(Chroma.from_texts)(texts, embeddings, metadatas=metadatas)
    # message_history = ChatMessageHistory()
    
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     output_key="answer",
    #     chat_memory=message_history,
    #     return_messages=True)
  
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        # memory=memory,
        output_key="answer",
        return_source_documents=False 
        )

    # Wrap the chain with message history capability
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return chain_with_history


def get_session_history(session_id: str):
    """Retrieves the chat history for a given session ID."""
    if not cl.user_session.get("chat_history"):
        cl.user_session.set("chat_history", ChatMessageHistory())
    return cl.user_session.get("chat_history")
