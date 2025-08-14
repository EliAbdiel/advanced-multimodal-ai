async def command_list() -> list:
    """Returns a list of available commands with their details."""
    return [
        {
            "id": "Picture", 
            "icon": "image", 
            "description": "Genera una imagen de un dragon"
        },
        {
            "id": "Scrape", 
            "icon": "file-code-2", 
            "description": "Extraer contenido de El País https://elpais.com"
        },
        {
            "id": "Search", 
            "icon": "globe", 
            "description": "Busca informacion sobre IA"
        },
        {
            "id": "Chat", 
            "icon": "message-square", 
            "description": "Escribe un codigo en Python"
        },
        {
            "id": "YouTube", 
            "icon": "video", 
            "description": "Transcribir video de YouTube https://www.youtube.com/watch?v="
        },
    ]