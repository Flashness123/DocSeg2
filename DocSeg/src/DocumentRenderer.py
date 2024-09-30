import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import chardet
import config
class DocumentRenderer:
    def __init__(self, original_image_path, docname, output_dir="./data/output"):
        self.original_image_path = original_image_path
        self.docname = docname
        self.base_path = Path(output_dir) / docname

    def read_text_file(self, file_path):
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        # Open the file with the detected encoding
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read().strip()

    def read_metadata(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_files(self, folder, txt_prefix, json_prefix):
        txt_files = []
        json_files = []

        folder_path = Path(f'data/output/{self.docname}/{folder}')

        # Look for files with the given prefixes
        txt_pattern = f"{txt_prefix}*.txt"  # Any number or text after the prefix
        json_pattern = f"{json_prefix}*.json"

        # Find all text and json files
        txt_files = list(folder_path.glob(txt_pattern))
        json_files = list(folder_path.glob(json_pattern))

        # Ensure both text and json files exist
        if not txt_files or not json_files:
            print(f"No matching files in {folder} for prefix: {txt_prefix}")

        return txt_files, json_files

    def collect_text_and_bounding_boxes(self):
        """
        Collect text and bounding box data from images, tables, and text files.
        """
        print("Starting collection of text and bounding boxes...")
        elements = []

        # Define folder and filename patterns
        folders = ['images', 'tables', 'texts']
        suffixes = [
            ('extracted_P', 'metadata_P'),
            ('df_extracted_', 'metadata_'),
            ('extracted_', 'metadata_')
        ]

        for folder, (txt_suffix, json_suffix) in zip(folders, suffixes):
            print(f"Processing folder: {folder}")

            # Get the text and json files for the current folder
            txt_files, json_files = self.get_files(folder, txt_suffix, json_suffix)

            print(f"Found {len(txt_files)} text files and {len(json_files)} JSON files in {folder}")

            for txt_file, json_file in zip(txt_files, json_files):
                print(f"Processing text file: {txt_file} and metadata file: {json_file}")

                # Read text content
                text = self.read_text_file(txt_file)

                # Read JSON metadata
                metadata = self.read_metadata(json_file)

                # Check if metadata contains a valid bounding box and text
                if metadata and 'normalized_bounding_box' in metadata and text:
                    bounding_boxes = metadata['normalized_bounding_box']

                    # Append the result if both text and bounding boxes are present
                    elements.append({
                        "text": text,
                        "bounding_boxes": bounding_boxes,
                        "label": metadata.get("label", "Unknown")
                    })
                    print(f"Collected element: text length = {len(text)}, bounding_boxes = {bounding_boxes}")
                else:
                    print(f"No valid text or bounding boxes in {txt_file} or {json_file}")

        print(f"Collection complete. Total elements: {len(elements)}")
        return elements

    def create_image_with_text(self, output_image_path, elements):
        """
        Create an image with text and bounding boxes drawn on it.

        Args:
        - output_image_path (str): Path to save the output image.
        - elements (list): List of elements containing text and bounding boxes.
        """
        # Load the original image to get its size
        original_image = Image.open(self.original_image_path)
        canvas_size = original_image.size  # Get the original size

        # Create a new image to draw on, with a white background
        output_image = Image.new("RGB", canvas_size, "white")
        # output_image.paste(original_image, (0, 0))  # Paste the original image onto the new canvas

        draw = ImageDraw.Draw(output_image)

        for element in elements:
            text = element['text']
            bbox = element['bounding_boxes']  # Single bounding box for the element

            # Extract the bounding box coordinates directly
            x1, y1, x2, y2 = bbox  # normalized (0 to 1) values

            # Convert normalized coordinates to pixel values
            x1 *= canvas_size[0]
            y1 *= canvas_size[1]
            x2 *= canvas_size[0]
            y2 *= canvas_size[1]

            # Draw the bounding box for visibility (optional)
            draw.rectangle([x1, y1, x2, y2], outline="black")

            # Fit the text within the bounding box
            max_width = x2 - x1
            max_height = y2 - y1

            # Create a font with a reasonable size
            font_size = 20  # Initial font size
            try:
                if config.FONT_PATH:
                    font = ImageFont.truetype(config.FONT_PATH, font_size)
                else:
                    font = ImageFont.load_default()
            except Exception as e:
                print(f"Failed to load font from {config.FONT_PATH}. Error: {e}. Using default font.")
                font = ImageFont.load_default()

            while True:
                # Measure text size
                # text_width, text_height = draw.textsize(text, font=font)
                _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
                if text_width <= max_width and text_height <= max_height:
                    break
                # Reduce font size and try again
                font_size -= 1
                try:
                    if config.FONT_PATH:
                        font = ImageFont.truetype(config.FONT_PATH, font_size)
                    else:
                        font = ImageFont.load_default()
                except Exception as e:
                    print(f"Failed to load font from {config.FONT_PATH}. Error: {e}. Using default font.")
                    font = ImageFont.load_default()
                if font_size <= 8:
                    break  # Avoid going too small


            is_table = element.get("label") == "Table"

            if is_table:
                # For table elements, draw the scaled text directly without wrapping
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    # Center the text vertically within the bounding box
                    #text_y = y1 + (max_height - text_height) / 2
                    draw.text((x1, y1), text, font=font, fill="black")
            else:
                # For non-table elements, allow wrapping
                lines = []
                words = text.split(' ')
                current_line = ""

                for word in words:
                    # Check if adding the next word exceeds the width
                    test_line = f"{current_line} {word}".strip()
                    #test_width, _ = draw.textsize(test_line, font=font)
                    _, _, test_width, _ = draw.textbbox((0, 0), test_line, font=font)

                    if test_width <= max_width:
                        current_line = test_line  # Append to the current line
                    else:
                        lines.append(current_line)  # Save the current line
                        current_line = word  # Start a new line with the current word

                # Add the last line if there's any text left
                if current_line:
                    lines.append(current_line)

                # Draw the lines of text inside the bounding box
                for i, line in enumerate(lines):
                    # Calculate the vertical position for each line
                    line_y = y1 + i * (font_size + 2)  # Add a small padding
                    if line_y + font_size <= y2:  # Ensure we don't go out of the bounding box
                        draw.text((x1, line_y), line, font=font, fill="black")

        # Save the final output image
        output_image.save(output_image_path)

    def run(self):

        elements = self.collect_text_and_bounding_boxes()

        if not elements:
            print("No elements to render. Please check the input files.")
            return

        output_image_path = self.base_path / "translated_document_visualization.png"
        self.create_image_with_text(output_image_path, elements)

        print(f"Document saved at {output_image_path}")


"""If this DocumentRenderer doesnt work as expected try using the one below:

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class DocumentRenderer:
    def __init__(self, original_image_path, docname, output_dir="./output"):
        self.original_image_path = original_image_path
        self.docname = docname
        self.base_path = Path(output_dir) / docname

        # Specify a larger font size (adjust the size as needed)
        self.font_size = 20  # Change this value for larger or smaller text

    def render_document(self):
        # Load the original image to get its size
        with Image.open(self.original_image_path) as img:
            img_width, img_height = img.size

        # Create a blank image with the same size
        blank_image = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(blank_image)

        # Render tables
        self._render_items(self.base_path / 'tables', draw, blank_image,
                           "metadata_Table_", "df_extracted_Table_")
        # Render texts
        self._render_items(self.base_path / 'texts', draw, blank_image,
                           "metadata_", "extracted_")
        # Render images
        self._render_items(self.base_path / 'images', draw, blank_image,
                           "metadata_", "extracted_")

        # Save the rendered image
        output_image_path = self.base_path / f"{self.docname}_rendered.png"
        blank_image.save(output_image_path)
        print(f"Rendered document saved as: {output_image_path}")

    def _render_items(self, folder, draw, image, metadata_prefix, text_prefix):
        for json_file in folder.glob(f"{metadata_prefix}*.json"):
            with open(json_file) as f:
                metadata = json.load(f)

            # Extract the relevant suffix from the JSON filename
            suffix = json_file.stem.split('_')[-1]
            # Determine the corresponding text file by matching patterns
            txt_files = list(folder.glob(f"{text_prefix}*{suffix}.txt"))

            # Check if any matching text files are found
            if not txt_files:
                print(f"No matching text file found for: {json_file}")
                continue

            # Assuming we take the first matching text file
            txt_file = txt_files[0]

            with open(txt_file, 'r', encoding='latin-1') as f:
                text = f.read()

            # Get normalized bounding box
            nbb = metadata.get("normalized_bounding_box", [])
            if len(nbb) != 4:
                print(f"Invalid bounding box for: {json_file}")
                continue

            # Calculate coordinates
            x_min = int(nbb[0] * image.width)
            y_min = int(nbb[1] * image.height)
            x_max = int(nbb[2] * image.width)
            y_max = int(nbb[3] * image.height)

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

            # Draw larger text inside the bounding box
            self._draw_larger_text(draw, text, (x_min + 5, y_min + 5))

    def _draw_larger_text(self, draw, text, position):
        # Load a default font or specify a path to a .ttf font file
        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)  # Specify font and size
        except IOError:
            font = ImageFont.load_default()  # Fallback to default font if specified font is not found

        # Draw the text
        draw.text(position, text, fill="black", font=font)  """