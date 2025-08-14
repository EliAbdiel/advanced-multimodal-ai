import io
import aiofiles
import chainlit as cl
import markdown

from weasyprint import HTML
from src.utils.config import EXTRACTED_DATA

async def content_as_pdf(content: str):
    """Generates a PDF from the provided markdown content."""
    # output_dir = Path("extracted_data")
    # output_dir.mkdir(exist_ok=True)

    pdf_bytes = await cl.make_async(_generate_pdf_bytes)(markdown_content=content)

    pdf_file_path = EXTRACTED_DATA / 'research_report.pdf'

    async with aiofiles.open(pdf_file_path, mode="wb") as f:
        await f.write(pdf_bytes)

    return str(pdf_file_path)

def _generate_pdf_bytes(markdown_content: str) -> bytes:
    """Generates PDF bytes from markdown content."""
    html_content = markdown.markdown(markdown_content)
    pdf_file = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_file)
    pdf_file.seek(0)
    return pdf_file.getvalue()