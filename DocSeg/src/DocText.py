import os
from pathlib import Path

import pytesseract
import easyocr
from PIL import Image

import config


class DocText:
    def __init__(self, ocr_type='tesseract', language='en', docname='STANDART'):
        """
        Initialize the TextExtractor with a choice of OCR engine.

        :param ocr_type: Choose between 'easyocr' or 'tesseract'.
        :param language: Language to use for OCR. Defaults to 'en' for English.
        """
        self.docname = docname
        self.ocr_type = ocr_type.lower()
        self.language = language
        if self.ocr_type == 'easyocr':
            self.reader = easyocr.Reader([language])
        if self.ocr_type == 'tesseract':
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

    def extract_text(self, image_path):
        """
        Extract text from the given image using the chosen OCR method.

        :param image_path: Path to the image file.
        :return: Extracted text as a string.
        """
        if self.ocr_type == 'easyocr':
            #image = Image.open(image_path)
            return "\n".join(self.reader.readtext(str(image_path), detail=0))
        elif self.ocr_type == 'tesseract':
            image = Image.open(image_path)  # Open the image using PIL
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            text = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Filter out low-confidence results
                    text.append(data['text'][i])
            return " ".join(text)  # Join the extracted words into a full text block
        else:
            raise ValueError("Invalid OCR type. Choose 'easyocr' or 'tesseract'.")

    # def perform_easyocr(self, image_path):
    #     result = self.reader.readtext(image_path, detail=0)  # detail=0 returns just the text
    #     return "\n".join(result)

    # def perform_tesseract(self, image_path):
    #     custom_config = r'--oem 3 --psm 6'  # Can be adjusted as needed
    #     text = pytesseract.image_to_string(image_path, config=custom_config, lang=self.language)
    #     return text

    def run(self):
        """
        Run the OCR on each image in the fixed directory ./output/{docname}/texts and save the text.
        """
        # Define the base input directory: ./output/{docname}/texts
        image_dir = Path(f'data/output/{self.docname}/texts')
        base_output_path = image_dir
        #base_output_path.mkdir(parents=True, exist_ok=True)

        # Loop over all images in the directory
        for image_path in image_dir.glob('*.png'):  # Assuming images are in PNG format
            try:
                extracted_text = self.extract_text(image_path)
                if extracted_text:
                    # Generate the text file path with the same base name as the image
                    text_file_path = base_output_path.joinpath(f"{image_path.stem}.txt")

                    # Save the extracted text to the file
                    with open(text_file_path, 'w', encoding='utf-8') as file:
                        file.write(extracted_text)

                    print(f"Text saved to {text_file_path}")
                else:
                    print(f"Failed to extract text from {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")