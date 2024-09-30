import openai
import base64
from pathlib import Path
import chardet
from PIL import Image

class DocImage:
    def __init__(self, docname, openai_api_key):
        self.docname = docname
        openai.api_key = openai_api_key  # Set the OpenAI API key

    def get_image_files(self):
        return list(Path(f'./output/{self.docname}/images').glob("*.png"))

    def read_text_files(self):
        all_texts = ""

        # Read from 'tables' folder
        tables_folder = Path(f'./output/{self.docname}/tables')
        for text_file in tables_folder.glob("*.txt"):
            with open(text_file, 'rb') as f:
                result = chardet.detect(f.read())

            with open(text_file, "r", encoding=result['encoding']) as f:
                all_texts += f.read() + "\n\n"

        # Read from 'texts' folder
        texts_folder = Path(f'./output/{self.docname}/texts')
        for text_file in texts_folder.glob("*.txt"):
            with open(text_file, 'rb') as f:
                result = chardet.detect(f.read())

            with open(text_file, "r", encoding=result['encoding']) as f:
                all_texts += f.read() + "\n\n"

        return all_texts.strip()

    def analyze_image(self, image_path):
        # Read the image and convert it to base64
        with open(image_path, "rb") as image_file:
            img_b64_str = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the prompt and image type
        img_type = "image/png"  # or "image/jpeg" depending on your image format
        prompt = "Was wird in dem Bild dargestellt?"

        # Read text data to use as context
        context_text = self.read_text_files()

        # Call the OpenAI API with context
        # response = openai.ChatCompletion.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": prompt},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
        #                 },
        #                 {"type": "text", "text": f"\nKontext text welcher sich in der Umgebung des Bildes befindet:\n{context_text}"}
        #             ],
        #         }
        #     ],
        # )
        #
        # return response.choices[0].message['content']
        # code for newer openai version
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                        },
                        {"type": "text",
                         "text": f"\nKontext text welcher sich in der Umgebung des Bildes befindet:\n{context_text}"}
                    ],
                }
            ],
        )

        return response.choices[0].message.content

    def save_analysis(self, image_path, analysis_result):
        """
        Save the analysis result as a .txt file next to the image file.

        Args:
        - image_path (Path): Path to the image file.
        - analysis_result (str): The result from the OpenAI API to save.
        """
        # Create the .txt file path next to the image
        output_path = image_path.with_suffix(".txt")

        # Write the result to the file
        with open(output_path, "w", encoding="utf-8") as text_file:
            text_file.write(analysis_result)

        print(f"Analysis saved to: {output_path}")

    def run(self):
        """
        Run the image analysis for all images in the image directory.
        """
        print(f"...Analyzing images from {self.docname}...")

        # Get all image files
        image_files = self.get_image_files()

        # Loop through all the images and analyze each one
        for image_path in image_files:
            print(f"Analyzing image: {image_path}")
            result = self.analyze_image(image_path)
            self.save_analysis(image_path, result)
            # print(f"Result for {image_path}: {result}")
