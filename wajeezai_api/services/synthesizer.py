"""code to get final word and pdf"""
"""this have list[SlideFusion] as input should use an LLM to sytesize the final word and pdf"""
"""we get the final word then convert it to pdf using docx2pdf"""
from wajeezai_api.services.alignment import SlideFusion


class Syntesizer:
    @staticmethod
    def create_word_doc(slide_fusions: list[SlideFusion]) -> str:
        """
        Creates a Word document from the list of SlideFusion objects.
        Returns the path to the created Word document.
        """
        # Implement logic to create a Word document using python-docx
        # Save the document and return the file path
        pass

    @staticmethod
    def convert_word_to_pdf(word_path: str) -> str:
        """
        Converts a Word document to PDF using docx2pdf.
        Returns the path to the created PDF document.
        """
        # Implement logic to convert Word to PDF using docx2pdf
        # Save the PDF and return the file path
        pass