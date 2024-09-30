import os
from pathlib import Path

import easyocr
import numpy as np
import pytesseract
import torch
from transformers import TableTransformerForObjectDetection, DetrImageProcessor # DetrFeatureExtractor
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

import config


class DocTable:
    def __init__(self, docname, multiple_in_one_detection, create_noise):
        """
        Initialize the DocTable class.

        Args:
        - docname (str): The name of the document directory containing the tables.
        - detection (str): Should the image split potentionally in case of multiple tables to display them each on a single image
        - create_noise (str): Should noise around the table be generated? For now it generates a header and a footer which makes it easier for the model to extract the rows and the columns
        """

        self.docname = docname
        self.multiple_in_one_detection = multiple_in_one_detection
        self.create_noise = create_noise
        self.table_detect_threshold = 0.2
        self.structure_extract_threshold = 0.5
        self.table_folder = Path(f'data/output/{self.docname}/tables')

        # Initialize the feature extractor and model for table structure recognition
        #self.feature_extractor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        self.model_structure = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

        # Initialize the second model and feature extractor for multiple table detection
        self.model_detection = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.detection_feature_extractor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection", size={"longest_edge": 1024})
        self.feature_extractor = DetrImageProcessor()

    def get_table_images(self):
        """
        Get all table images from the 'tables' subfolder and detect multiple tables in a single image.

        Returns:
        - List of paths to extracted table images.
        """
        # Collect all images in the tables folder
        all_table_images = list(self.table_folder.glob("*.png"))
        extracted_tables = []
        if self.multiple_in_one_detection:
            # Loop through each table image
            for image_path in all_table_images:
                print(f"Processing potential multiple tables in: {image_path}")

                # Load the image
                image = Image.open(image_path).convert("RGB")
                width, height = image.size

                # Prepare the image for detection
                encoding = self.detection_feature_extractor(image, return_tensors="pt")

                # Detect tables using the table detection model
                with torch.no_grad():
                    outputs = self.model_detection(**encoding)

                # Post-process the detected results (tables)
                results = self.detection_feature_extractor.post_process_object_detection(
                    outputs, threshold=self.table_detect_threshold, target_sizes=[(height, width)]
                )[0]

                # Margin for cropping
                margin = 16

                # Loop through the detected tables, crop them, and save
                for idx, box in enumerate(results['boxes']):
                    box = [int(i) for i in box.tolist()]  # Convert the box to integers for cropping
                    box[0] = max(box[0] - margin, 0)  # Left
                    box[1] = max(box[1] - margin, 0)  # Top
                    box[2] = min(box[2] + margin, width)  # Right
                    box[3] = min(box[3] + margin, height)  # Bottom

                    # Crop the image using the bounding box
                    cropped_image = image.crop((box[0], box[1], box[2], box[3]))

                    # Save the cropped image
                    cropped_table_path = self.table_folder / f"table_{image_path.stem}_{idx + 1}.png"
                    cropped_image.save(cropped_table_path)
                    extracted_tables.append(cropped_table_path)

                    print(f"Saved {cropped_table_path}")

            return extracted_tables
        else:
            return all_table_images

    def process_image(self, image_path):
        """
        Process a single table image using the transformer model.

        Args:
        - image_path (str): Path to the image file.

        Returns:
        - Processed results from the model including bounding boxes and labels.
        """
        image = Image.open(image_path).convert("RGB")
        if self.create_noise:
            header_text = "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit."
            footer_text = "Neque porro quisquam est qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit."
            font_size = 30  # Adjust this value to change the text size
            try:
                if config.FONT_PATH:
                    print('Config.FONT_PATH is True')
                    font = ImageFont.truetype(config.FONT_PATH, font_size)
                else:
                    print('Config.FONT_PATH is False')
                    font = ImageFont.load_default()
            except Exception as e:
                print(f"Failed to load font from {config.FONT_PATH}. Error: {e}. Using default font.")
                font = ImageFont.load_default()

            # Create the header image (width same as table, height depending on font size)
            header_height = font_size + 20
            header_image = Image.new("RGB", (image.width, header_height), "white")
            draw = ImageDraw.Draw(header_image)
            # text_width, text_height = draw.textsize(header_text, font=font)
            _, _, text_width, text_height = draw.textbbox((0, 0), header_text, font=font)
            draw.text(((header_image.width - text_width) / 2, (header_height - text_height) / 2), header_text,
                      font=font, fill="black")

            # Create the footer image (width same as table, height depending on font size)
            footer_height = font_size + 20
            footer_image = Image.new("RGB", (image.width, footer_height), "white")
            draw = ImageDraw.Draw(footer_image)
            # text_width, text_height = draw.textsize(footer_text, font=font)
            _, _, text_width, text_height = draw.textbbox((0, 0), footer_text, font=font)
            draw.text(((footer_image.width - text_width) / 2, (footer_height - text_height) / 2), footer_text,
                      font=font, fill="black")

            # Combine header, table image, and footer
            total_height = header_height + image.height + footer_height
            combined_image = Image.new("RGB", (image.width, total_height), "white")

            # Paste header, table, and footer images
            combined_image.paste(header_image, (0, 0))
            combined_image.paste(image, (0, header_height))
            combined_image.paste(footer_image, (0, header_height + image.height))
            # image_path = Path(self.table_folder) / f"{Path(image_path).stem}_header_footer.png" # ONLY new table gets saved so that tables dont add up in folder, otherwise for each iteration EVERY table gets analyzed aand saved in same fodler again...
            combined_image.save(image_path)
            image = combined_image

        # Load and preprocess the image

        width, height = image.size
        encoding = self.feature_extractor(image, return_tensors="pt")

        # Run the model to get the outputs
        with torch.no_grad():
            outputs = self.model_structure(**encoding)

        # Post-process the results to get bounding boxes and labels
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=self.structure_extract_threshold, target_sizes=[(height, width)])[0]
        return image, results, image_path

    def plot_results_specific(self, image, scores, labels, boxes, labelnums):
        """
        Plot the image with bounding boxes for specific labels.

        Args:
        - image: The PIL image object.
        - scores: The scores for the detected objects.
        - labels: The labels for the detected objects.
        - boxes: The bounding boxes for the detected objects.
        - labelnums: A list of label numbers to highlight.

        Returns:
        - None: The function displays the image with the bounding boxes drawn.
        """
        # Convert the image to a format suitable for plotting
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        # Draw bounding boxes for all specified labels
        for score, label, box in zip(scores, labels, boxes):
            if label.item() in labelnums:
                xmin, ymin, xmax, ymax = box.tolist()
                # Draw the bounding box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)
                # Annotate the label and score
                ax.text(xmin, ymin, f'{label.item()}:{score:.2f}', color='blue', fontsize=12,
                        bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

    def draw_box_specific(self, image, labelnum):
        image = image.convert("RGB")
        width, height = image.size

        encoding = self.feature_extractor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model_structure(**encoding)

        results = \
        self.feature_extractor.post_process_object_detection(outputs, threshold=self.structure_extract_threshold, target_sizes=[(height, width)])[
            0]  # aryn uses 0.32
        self.plot_results_specific(image, results['scores'], results['labels'], results['boxes'], labelnum)

    def build_table_dataframe(self, results):
        """
        Build a table structure using rows, columns, headers, and colspans from the results.

        Args:
        - results (dict): The results containing labels, boxes, and scores from the table structure detection model.

        Returns:
        - coordinate_df (pd.DataFrame): A DataFrame representing the table structure.
        """
        labels = results['labels']
        boxes = results['boxes']
        scores = results['scores']

        # Initialize the grid dimensions based on labels
        n_cols = (labels == 1).sum().item()
        n_rows = (labels == 2).sum().item()

        print(f"1. n_cols: {n_cols}, n_rows: {n_rows}")

        # Identify all header and colspan cells
        header_boxes = [boxes[i] for i, label in enumerate(labels) if label.item() == 3]
        colspan_boxes = [boxes[i] for i, label in enumerate(labels) if label.item() == 5]

        # Find the uppermost rows and leftmost columns
        rows = [(box[1].item(), box[3].item(), i) for i, (label, box) in enumerate(zip(labels, boxes)) if
                label.item() == 2]
        cols = [(box[0].item(), box[2].item(), i) for i, (label, box) in enumerate(zip(labels, boxes)) if
                label.item() == 1]

        # Sort rows and columns by their top and left coordinates
        rows.sort()
        cols.sort()

        # Adjust the rows so they don't overlap and perfectly align
        for i in range(len(rows) - 1):
            row_top, row_bottom, row_i = rows[i]
            next_row_top, next_row_bottom, next_row_i = rows[i + 1]
            # Ensure no overlap and no gap
            if row_bottom > next_row_top:
                midpoint = (row_bottom + next_row_top) / 2
                rows[i] = (row_top, midpoint, row_i)
                rows[i + 1] = (midpoint, next_row_bottom, next_row_i)
            else:
                rows[i] = (row_top, next_row_top, row_i)

        # Adjust the columns so they don't overlap and perfectly align
        for i in range(len(cols) - 1):
            col_left, col_right, col_i = cols[i]
            next_col_left, next_col_right, next_col_i = cols[i + 1]
            # Ensure no overlap and no gap
            if col_right > next_col_left:
                midpoint = (col_right + next_col_left) / 2
                cols[i] = (col_left, midpoint, col_i)
                cols[i + 1] = (midpoint, next_col_right, next_col_i)
            else:
                cols[i] = (col_left, next_col_left, col_i)

        # Initialize the grid with the new adjusted rows and columns
        grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]

        # Function to calculate overlap ratio
        def calculate_overlap(box1, box2):
            x1_max = max(box1[0], box2[0])
            y1_max = max(box1[1], box2[1])
            x2_min = min(box1[2], box2[2])
            y2_min = min(box1[3], box2[3])

            overlap_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

            return overlap_area / min(box1_area, box2_area)
            return 0

        # Track unique colspan numbers
        colspan_counter = 0
        colspan_map = {}
        merged_colspan_boxes = {}

        # Fill the grid with adjusted row and column boxes and include metadata
        for row_idx, (row_top, row_bottom, row_i) in enumerate(rows):
            row_box = boxes[row_i]
            for col_idx, (col_left, col_right, col_i) in enumerate(cols):
                col_box = boxes[col_i]
                overlap_top = max(row_top, col_box[1].item())
                overlap_bottom = min(row_bottom, col_box[3].item())
                overlap_left = max(col_left, row_box[0].item())
                overlap_right = min(col_right, row_box[2].item())

                if overlap_top < overlap_bottom and overlap_left < overlap_right:
                    # Create the box for the current cell
                    cell_box = [overlap_left, overlap_top, overlap_right, overlap_bottom]

                    # Determine if this cell is a header or colspan by checking overlaps
                    is_header = any(calculate_overlap(cell_box, header_box) > 0.5 for header_box in header_boxes)
                    is_colspan = any(calculate_overlap(cell_box, colspan_box) > 0.5 for colspan_box in colspan_boxes)

                    colspan_number = None
                    if is_colspan:
                        # Check if this colspan already has a number, otherwise assign a new one
                        overlapping_colspan_boxes = [box for box in colspan_boxes if
                                                     calculate_overlap(cell_box, box) > 0.5]
                        for box in overlapping_colspan_boxes:
                            box_tuple = tuple(box.tolist())
                            if box_tuple in colspan_map:
                                colspan_number = colspan_map[box_tuple]
                                break

                        if colspan_number is None:
                            colspan_counter += 1
                            colspan_number = colspan_counter
                            for box in overlapping_colspan_boxes:
                                box_tuple = tuple(box.tolist())
                                colspan_map[box_tuple] = colspan_number

                        # Merge the bounding boxes for the same colspan
                        if colspan_number not in merged_colspan_boxes:
                            merged_colspan_boxes[colspan_number] = cell_box
                        else:
                            merged_box = merged_colspan_boxes[colspan_number]
                            merged_colspan_boxes[colspan_number] = [
                                min(merged_box[0], cell_box[0]),  # x1
                                min(merged_box[1], cell_box[1]),  # y1
                                max(merged_box[2], cell_box[2]),  # x2
                                max(merged_box[3], cell_box[3])  # y2
                            ]

        # Apply the merged boxes to all relevant cells
        for row_idx, (row_top, row_bottom, row_i) in enumerate(rows):
            row_box = boxes[row_i]
            for col_idx, (col_left, col_right, col_i) in enumerate(cols):
                col_box = boxes[col_i]
                overlap_top = max(row_top, col_box[1].item())
                overlap_bottom = min(row_bottom, col_box[3].item())
                overlap_left = max(col_left, row_box[0].item())
                overlap_right = min(col_right, row_box[2].item())

                if overlap_top < overlap_bottom and overlap_left < overlap_right:
                    # Create the box for the current cell
                    cell_box = [overlap_left, overlap_top, overlap_right, overlap_bottom]

                    # Determine if this cell is a header or colspan by checking overlaps
                    is_header = any(calculate_overlap(cell_box, header_box) > 0.5 for header_box in header_boxes)
                    is_colspan = any(calculate_overlap(cell_box, colspan_box) > 0.5 for colspan_box in colspan_boxes)

                    colspan_number = None
                    if is_colspan:
                        # Check if this colspan already has a number, otherwise assign a new one
                        overlapping_colspan_boxes = [box for box in colspan_boxes if
                                                     calculate_overlap(cell_box, box) > 0.5]
                        for box in overlapping_colspan_boxes:
                            box_tuple = tuple(box.tolist())
                            if box_tuple in colspan_map:
                                colspan_number = colspan_map[box_tuple]
                                break

                    # Store the merged box in the grid with metadata
                    cell_data = {
                        'box': merged_colspan_boxes.get(colspan_number, cell_box) if is_colspan else cell_box,
                        'is_header': is_header,
                        'is_colspan': (is_colspan, colspan_number) if is_colspan else False,
                        'score': scores[row_i].item()
                    }
                    grid[row_idx][col_idx] = cell_data

        # Crop out rows that are completely None
        grid = [row for row in grid if any(cell is not None for cell in row)]
        n_rows = len(grid)  # Update the number of rows

        # Transpose and crop columns that are completely None
        grid = list(map(list, zip(*grid)))  # Transpose the grid to handle columns as rows
        grid = [col for col in grid if any(cell is not None for cell in col)]
        n_cols = len(grid)  # Update the number of columns
        grid = list(map(list, zip(*grid)))  # Transpose back to original orientation

        print(f"2. n_cols: {n_cols}, n_rows: {n_rows}")

        # Convert the grid to a DataFrame
        coordinate_df = pd.DataFrame(grid)

        # Configure Pandas display options
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Auto adjust the width
        pd.set_option('display.max_colwidth', None)  # Show full column content

        return coordinate_df

    def extract_text_from_table_tesseract(self, image_path, grid, n_cols, n_rows):
        """
        Extract text using Tesseract and map it to the corresponding grid cells.

        Args:
        - image_path (str): The path to the table image.
        - grid (list): The table grid with bounding boxes.
        - n_cols (int): The number of columns in the table.
        - n_rows (int): The number of rows in the table.

        Returns:
        - word_df (pd.DataFrame): A DataFrame representing the extracted words in the table cells.
        """
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Perform OCR using Tesseract
        results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Initialize the word grid
        word_grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]

        n_boxes = len(results['text'])
        for i in range(n_boxes):
            word = results['text'][i].strip()
            if word == "":
                continue  # Skip empty text results

            # Get bounding box coordinates
            xmin = results['left'][i]
            ymin = results['top'][i]
            xmax = xmin + results['width'][i]
            ymax = ymin + results['height'][i]

            # Calculate the center of the word's bounding box
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Find the corresponding cell in the grid based on the center of the word's bounding box
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    cell_data = grid[row_idx][col_idx]
                    if cell_data:
                        grid_xmin, grid_ymin, grid_xmax, grid_ymax = cell_data['box']
                        # Check if the center of the word's bounding box is within the grid cell's bounding box
                        if grid_xmin <= x_center <= grid_xmax and grid_ymin <= y_center <= grid_ymax:
                            # Add the word to the current cell if it's not already present
                            if word not in word_grid[row_idx][col_idx]:
                                word_grid[row_idx][col_idx] += word + " "

                            # If the cell is part of a colspan, add the word to all cells in that colspan
                            if cell_data['is_colspan']:
                                colspan_number = cell_data['is_colspan'][1]
                                for r in range(n_rows):
                                    for c in range(n_cols):
                                        if grid[r][c] and grid[r][c]['is_colspan'] and grid[r][c]['is_colspan'][
                                            1] == colspan_number:
                                            # Add the word only if it's not already in the cell
                                            if word not in word_grid[r][c]:
                                                word_grid[r][c] += word + " "
                            break  # Assuming a word fits in only one grid cell

        # Convert the word grid to a DataFrame
        word_df = pd.DataFrame(word_grid)

        # Display the word grid
        return word_df

    def extract_text_from_table_easyocr(self, image_path, grid, n_cols, n_rows):
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])  # Specify the language(s) you want to use

        # Load the image using PIL
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Use EasyOCR to extract words and their bounding boxes
        results = reader.readtext(image_np, detail=1)  # detail=1 provides bounding boxes and text

        # Assuming the grid and boxes are already defined from the previous steps:
        # n_cols, n_rows, and grid are defined as before.

        # Initialize the word grid
        word_grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]

        # Go through each detected word and its bounding box
        for result in results:
            bbox, word, _ = result
            (xmin, ymin), (xmax, ymax) = bbox[0], bbox[2]

            # Calculate the center of the word's bounding box
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Find the corresponding cell in the grid based on the center of the word's bounding box
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    cell_data = grid[row_idx][col_idx]
                    if cell_data:
                        grid_xmin, grid_ymin, grid_xmax, grid_ymax = cell_data['box']
                        # Check if the center of the word's bounding box is within the grid cell's bounding box
                        if grid_xmin <= x_center <= grid_xmax and grid_ymin <= y_center <= grid_ymax:
                            # Add the word to the current cell if it's not already present
                            if word not in word_grid[row_idx][col_idx]:
                                word_grid[row_idx][col_idx] += word + " "

                            # If the cell is part of a colspan, add the word to all cells in that colspan
                            if cell_data['is_colspan']:
                                colspan_number = cell_data['is_colspan'][1]
                                for r in range(n_rows):
                                    for c in range(n_cols):
                                        if grid[r][c] and grid[r][c]['is_colspan'] and grid[r][c]['is_colspan'][
                                            1] == colspan_number:
                                            # Add the word only if it's not already in the cell
                                            if word not in word_grid[r][c]:
                                                word_grid[r][c] += word + " "
                            break  # Assuming a word fits in only one grid cell

        # Convert the word grid to a DataFrame
        word_df = pd.DataFrame(word_grid)

        # Display the word grid
        return word_df

    def generate_tabletext_output(self, table_image, grid, n_cols, n_rows, delimiter=',,'):
        """
        Generate a table output text with metadata like row, column, word, header status, colspan, etc.

        Args:
        - image_path (str): The path to the table image.
        - grid (list): The table grid with bounding boxes.
        - n_cols (int): Number of columns.
        - n_rows (int): Number of rows.
        - delimiter (str): Delimiter to separate table metadata.

        Returns:
        - tabletext_output (str): The final text output for the table.
        """

        image = Image.open(table_image).convert("RGB")
        results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        word_grid = [[{"word": "", "is_header": False, "is_colspan": False, "colspan_number": None, "row": row_idx,
                       "col": col_idx}
                      for col_idx in range(n_cols)] for row_idx in range(n_rows)]
        # Go through each detected word and its bounding box
        n_boxes = len(results['text'])
        for i in range(n_boxes):
            word = results['text'][i].strip()
            if word == "":
                continue  # Skip empty text results

            # Get bounding box coordinates
            xmin = results['left'][i]
            ymin = results['top'][i]
            xmax = xmin + results['width'][i]
            ymax = ymin + results['height'][i]

            # Calculate the center of the word's bounding box
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Find the corresponding cell in the grid based on the center of the word's bounding box
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    cell_data = grid[row_idx][col_idx]
                    if cell_data:
                        grid_xmin, grid_ymin, grid_xmax, grid_ymax = cell_data['box']
                        # Check if the center of the word's bounding box is within the grid cell's bounding box
                        if grid_xmin <= x_center <= grid_xmax and grid_ymin <= y_center <= grid_ymax:
                            # Add the word to the current cell if it's not already present
                            if word not in word_grid[row_idx][col_idx]["word"]:
                                word_grid[row_idx][col_idx]["word"] += word + " "

                            # Add metadata to the word grid cell
                            word_grid[row_idx][col_idx]["is_header"] = cell_data['is_header']
                            word_grid[row_idx][col_idx]["is_colspan"] = bool(cell_data['is_colspan'])

                            # Check if this cell is part of a colspan, and extract the corresponding colspan number
                            if cell_data['is_colspan']:
                                colspan_number = cell_data['is_colspan'][1]
                                word_grid[row_idx][col_idx]["colspan_number"] = colspan_number

                                # Ensure all cells in the colspan have the correct metadata
                                for r in range(n_rows):
                                    for c in range(n_cols):
                                        if grid[r][c] and grid[r][c]['is_colspan'] and grid[r][c]['is_colspan'][
                                            1] == colspan_number:
                                            # Add the word only if it's not already in the cell
                                            if word not in word_grid[r][c]["word"]:
                                                word_grid[r][c]["word"] += word + " "
                                            # Ensure both start and end cells in the colspan are set correctly
                                            word_grid[r][c]["is_header"] = cell_data['is_header']
                                            word_grid[r][c]["is_colspan"] = True
                                            word_grid[r][c]["colspan_number"] = colspan_number
                            break  # Assuming a word fits in only one grid cell

        # Prepare the final table output
        table_output = []

        # Flatten the word grid into a single output table
        for row_idx, row in enumerate(word_grid):
            row_data = []
            for col_idx, cell in enumerate(row):
                row_data.append(
                    f'row: {row_idx}{delimiter}col: {col_idx}{delimiter}word: {cell["word"].strip()}{delimiter}is_header: {cell["is_header"]}{delimiter}is_colspan: {cell["is_colspan"]}{delimiter}corresponding_colspan_number: {cell["colspan_number"]}'
                )
            table_output.append(delimiter.join(row_data))

        # Join rows with a newline character
        tabletext_output = "\n".join(table_output)

        # Optionally, save to a file
        # Create the path to save the output in the same folder as the table image
        table_name = Path(table_image).stem  # Extract table image name (without extension)
        output_file = Path(self.table_folder) / f"{table_name}_tabletext_output.txt"

        # Save the output to the appropriate folder
        with open(output_file, 'w') as f:
            f.write(tabletext_output)

    def save_as_csv(self, table_image_path_pretty, word_df):
        with open(table_image_path_pretty, 'a') as f:
            df_string = word_df.to_string(header=False, index=False)
            f.write(df_string)



    def run(self):
        print(f"...Program is starting to transform tables from {self.docname}...")
        table_images = self.get_table_images()
        #table_images = [r"C:\Users\lukask\Coding\Interface Projects\Segmentation\DocSeg\DocSeg\output\doc1\tables\NielsRogge_Image.png"]

        # Loop through all the table images and process each one
        for table_image_path in table_images:
            print(f"Processing table image: {table_image_path}")
            image, results, table_image_path = self.process_image(table_image_path)
            #self.draw_box_specific(image, [5])  # enable for showing rows, cols
            coordinate_df = self.build_table_dataframe(results)
            print(coordinate_df)
            word_df = self.extract_text_from_table_tesseract(table_image_path, grid=coordinate_df.values, n_cols=coordinate_df.shape[1],n_rows=coordinate_df.shape[0]) # also executable with easyocr
            print(word_df)
            base_path = table_image_path.with_suffix('')  # This removes the .png extension
            table_image_path_pretty = base_path.parent / f"df_{base_path.stem}.txt"  # Prefix pretty_ and create the new .txt filename

            self.save_as_csv(table_image_path_pretty, word_df)
            self.generate_tabletext_output(table_image_path, grid=coordinate_df.values, n_cols=coordinate_df.shape[1],n_rows=coordinate_df.shape[0], delimiter=',,')
