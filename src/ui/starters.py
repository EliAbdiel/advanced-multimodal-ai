import chainlit as cl

async def select_starter():
    """
    Returns a list of starter options for the chatbot interface.
    Each starter contains a label, message prompt, and associated icon.
    """
    return [
        cl.Starter(
            label="Ética IA Investigación",
            message="Busca información sobre la ética en la inteligencia artificial.",
            icon="/public/starters/search-globe.svg",
            ),
        cl.Starter(
            label="Genera una imagen",
            message="Genera un dibujo de un dragón volando.",
            icon="/public/starters/image-picture.svg",
            ),
        # cl.Starter(
        #     label="Set Small Goals",
        #     message="Do you have any tips for staying motivated?",
        #     icon="/public/starters/write.svg",
        #     ),
        cl.Starter(
            label="Transcribe Youtube video",
            message="Dame un resumen en 3 lineas del siguiente video https://youtu.be/rwF-X5STYks?si=gXd7O8e7q5NpHm8h",
            icon="/public/starters/youtube.svg",
            ),
        cl.Starter(
            label="Aprende machine learning",
            message="Recomiéndame algunos recursos para aprender sobre machine learning",
            icon="/public/starters/human-learn.svg",
            ),
        cl.Starter(
            label="Scrapear una página web",
            message="Resumir esta pagina de documentacion de LangChain en 3 lineas https://python.langchain.com/docs/tutorials/summarization/",
            icon="/public/starters/code-file.svg",
            ),
        cl.Starter(
            label="Escribe un código",
            message="Escribe un código en Python que imprima 'Hola, mundo!'",
            icon="/public/starters/python.svg",
            ),
        ]