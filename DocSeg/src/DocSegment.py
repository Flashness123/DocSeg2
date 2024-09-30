import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from timeit import default_timer as timer
import easyocr
import pytesseract
import json

class DocSegment:
    def __init__(self, path, docname = None, threshold=0.4, proximity_threshold=500, vertical_bias=100, iou_threshold=0.2):
        self.path = path
        self.docname = docname if docname else os.path.basename(path)
        self.threshold = threshold
        self.proximity_threshold = proximity_threshold
        self.vertical_bias = vertical_bias
        self.iou_threshold = iou_threshold
        self.processor = AutoImageProcessor.from_pretrained("Aryn/deformable-detr-DocLayNet")
        self.model = DeformableDetrForObjectDetection.from_pretrained("Aryn/deformable-detr-DocLayNet")
        self.reader = easyocr.Reader(['de'])

    def merge_bounding_boxes(self, results):
        def get_center(box):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            return np.array([x_center, y_center])

        def compute_distance(box1, box2):
            center1 = get_center(box1)
            center2 = get_center(box2)
            distance = np.linalg.norm(center1 - center2)
            if center2[1] > center1[1]:  # Prefer connections below
                distance -= self.vertical_bias
            return distance

        labels_to_merge = [1, 8]  # 'Caption', 'Section-header'
        nearby_labels = [10, 3, 7, 9]  # 'Text', 'Formula', 'Picture', 'Table'

        boxes_to_merge = []
        nearby_boxes = []
        other_boxes = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_id = label.item()
            if label_id in labels_to_merge:
                boxes_to_merge.append((label_id, box.tolist()))
            elif label_id in nearby_labels:
                nearby_boxes.append((label_id, box.tolist()))
            else:
                other_boxes.append((label_id, box.tolist()))

        new_boxes = []
        used_nearby_boxes = set()

        for label, box in boxes_to_merge:
            best_distance = float('inf')
            best_box = None
            best_label = None

            for nearby_label, nearby_box in nearby_boxes:
                distance = compute_distance(box, nearby_box)
                if distance < best_distance and distance < self.proximity_threshold and tuple(nearby_box) not in used_nearby_boxes:
                    best_distance = distance
                    best_box = nearby_box
                    best_label = nearby_label

            if best_box is not None:
                merged_box = [
                    min(box[0], best_box[0]), min(box[1], best_box[1]),
                    max(box[2], best_box[2]), max(box[3], best_box[3])
                ]
                new_boxes.append((best_label, merged_box))
                used_nearby_boxes.add(tuple(best_box))
            else:
                new_boxes.append((label, box))

        for label, box in nearby_boxes:
            if tuple(box) not in used_nearby_boxes:
                new_boxes.append((label, box))

        new_boxes.extend(other_boxes)

        new_bbox_results = {
            "boxes": torch.tensor([box for _, box in new_boxes]),
            "labels": torch.tensor([label for label, _ in new_boxes]),
            "scores": results["scores"],
        }

        return new_bbox_results

    def merge_overlapping_boxes(self, new_bbox_results):
        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            intersection_width = max(0, x2 - x1)
            intersection_height = max(0, y2 - y1)
            intersection_area = intersection_width * intersection_height

            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

            union_area = box1_area + box2_area - intersection_area

            iou = intersection_area / union_area if union_area != 0 else 0
            return iou

        def merge_boxes(box1, box2):
            x_min = min(box1[0], box2[0])
            y_min = min(box1[1], box2[1])
            x_max = max(box1[2], box2[2])
            y_max = max(box1[3], box2[3])
            return [x_min, y_min, x_max, y_max]

        bounding_boxes = new_bbox_results["boxes"].tolist()
        labels = new_bbox_results["labels"].tolist()
        scores = new_bbox_results["scores"].tolist()

        merged_boxes = []
        merged_labels = []
        merged_scores = []

        used_indices = set()

        for i in range(len(bounding_boxes)):
            if i in used_indices:
                continue

            box1 = bounding_boxes[i]
            label1 = labels[i]
            score1 = scores[i]

            for j in range(i + 1, len(bounding_boxes)):
                if j in used_indices:
                    continue

                box2 = bounding_boxes[j]
                iou = compute_iou(box1, box2)

                if iou > self.iou_threshold:
                    box1 = merge_boxes(box1, box2)
                    used_indices.add(j)

            merged_boxes.append(box1)
            merged_labels.append(label1)
            merged_scores.append(score1)

        merged_bbox_results = {
            "boxes": torch.tensor(merged_boxes),
            "labels": torch.tensor(merged_labels),
            "scores": torch.tensor(merged_scores),
        }

        return merged_bbox_results

    def draw_bounding_boxes_with_caption(self, image, results, caption):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        # text_width, text_height = draw.textsize(caption, font)
        _, _, text_width, text_height = draw.textbbox((0, 0), caption, font=font)
        draw.rectangle(((0, 0), (image.width, text_height + 4)), fill="black")
        draw.text((image.width / 2 - text_width / 2, 0), caption, fill="white", font=font)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()]
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{label_name} ({round(score.item(), 3)})", fill="red")

        return image

    def show_labels_before(self, image, results):
        print("Before merging is shown")
        before_image = image.copy()
        before_image_with_boxes = self.draw_bounding_boxes_with_caption(before_image, results, "Before Merging")
        before_image_with_boxes.show()

    def show_labels_after(self, image, new_bbox_results):
        print("After merging is shown")
        after_image = image.copy()
        after_image_with_boxes = self.draw_bounding_boxes_with_caption(after_image, new_bbox_results, "After Merging")
        after_image_with_boxes.show()

    def whiten_bounding_boxes(self, image, bounding_boxes):
        draw = ImageDraw.Draw(image)
        for box in bounding_boxes:
            left, top, right, bottom = box
            draw.rectangle([left, top, right, bottom], fill="white")
        return image

    def perform_ocr_easyocr(self, image):
        image_np = np.array(image)
        ocr_results = self.reader.readtext(image_np, detail=1)

        ocr_boxes = []
        for result in ocr_results:
            box = result[0]
            x_min = min(point[0] for point in box)
            y_min = min(point[1] for point in box)
            x_max = max(point[0] for point in box)
            y_max = max(point[1] for point in box)
            ocr_boxes.append([x_min, y_min, x_max, y_max])

        return ocr_boxes

    def perform_ocr_tesseract(self, image):
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Set Tesseract configuration to use OCR Engine Mode 3 and Page Segmentation Mode 11
        custom_config = r'--oem 3 --psm 11'

        # Perform OCR using Tesseract
        data = pytesseract.image_to_data(image_np, config=custom_config, output_type=pytesseract.Output.DICT)

        ocr_boxes = []

        # Extract bounding boxes for each word detected
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # You can set a minimum confidence threshold here if needed
                x_min = data['left'][i]
                y_min = data['top'][i]
                x_max = x_min + data['width'][i]
                y_max = y_min + data['height'][i]
                ocr_boxes.append([x_min, y_min, x_max, y_max])

        return ocr_boxes

    def merge_ocr_with_model_boxes(self, extracted_boxes, ocr_boxes, max_merges=1):  # , merge_proximity_threshold=50 implement that vertical goes before horizontal merge
        def is_neighbour(box1, box2):
            # Calculate horizontal and vertical distances between boxes
            horizontal_distance = max(0, max(box1[0], box2[0]) - min(box1[2], box2[2]))
            vertical_distance = max(0, max(box1[1], box2[1]) - min(box1[3], box2[3]))

            # If the distance between any two boxes is less than the threshold in both directions, consider them close
            return horizontal_distance <= self.threshold and vertical_distance <= self.threshold

        merged_boxes = extracted_boxes.copy()  # Start with the model-detected boxes
        used_ocr_indices = set()

        # Merge each OCR box with up to max_merges nearest model boxes
        for obox in ocr_boxes:
            distances = []
            for i, ebox in enumerate(merged_boxes):
                if is_neighbour(ebox, obox):
                    # Compute the distance between the centers of the boxes
                    ebox_center = [(ebox[0] + ebox[2]) / 2, (ebox[1] + ebox[3]) / 2]
                    obox_center = [(obox[0] + obox[2]) / 2, (obox[1] + obox[3]) / 2]
                    distance = np.linalg.norm(np.array(ebox_center) - np.array(obox_center))
                    distances.append((distance, i, ebox))

            # Sort by distance and select the closest up to max_merges boxes
            distances.sort(key=lambda x: x[0])
            closest_boxes = distances[:max_merges]

            for _, best_box_index, best_box in closest_boxes:
                if best_box is not None:
                    # Merge the OCR box with the selected model box
                    new_box = [
                        min(best_box[0], obox[0]), min(best_box[1], obox[1]),
                        max(best_box[2], obox[2]), max(best_box[3], obox[3])
                    ]
                    merged_boxes[best_box_index] = new_box
                    print("OCR Box got merged")
                    used_ocr_indices.add(tuple(obox))

        return merged_boxes, used_ocr_indices

    def save_extracted_boxes(self, image, merged_boxes, labels):
        extracted_images = []
        image_width, image_height = image.size

        #metadata = []  # for the bounding boxes

        # Create document-specific directories and get the base directory path
        base_dir = self.create_directory_structure()

        # Iterate through the boxes and their labels
        for i, (box, label) in enumerate(zip(merged_boxes, labels)):
            left, top, right, bottom = box
            extracted_image = image.crop((left, top, right, bottom))
            extracted_images.append(extracted_image)

            # Get the label name from the model configuration
            label_name = self.model.config.id2label[label]

            # Determine the appropriate directory based on the label
            if label_name in ["Formula", "Picture"]:
                save_dir = os.path.join(base_dir, 'images')
            elif label_name in ["Table"]:
                save_dir = os.path.join(base_dir, 'tables')
            elif label_name in ["Caption","Text", "Title", "List-item", "Footnote", "Page-footer", "Page-header", "Section-header"]:
                save_dir = os.path.join(base_dir, 'texts')
            else:
                save_dir = os.path.join(base_dir, 'unknown')  # Use 'vectors' for anything else

            # Make sure the directory exists (just in case)
            os.makedirs(save_dir, exist_ok=True)

            # Save the extracted image in the appropriate folder
            extracted_image_path = os.path.join(save_dir, f"extracted_{label_name}_{i}.png")
            extracted_image.save(extracted_image_path)
            print(f"Saved {label_name} to: {extracted_image_path}")

            normalized_bbox = self.normalize_bboxes([box], image_width, image_height)[0]

            # Save bounding box metadata as JSON alongside the extracted image
            metadata = {
                "label": label_name,
                "normalized_bounding_box": normalized_bbox,
                "image_path": extracted_image_path
            }

            metadata_file = os.path.join(save_dir, f"metadata_{label_name}_{i}.json")
            with open(metadata_file, 'w', encoding='utf-8') as file:
                json.dump(metadata, file, ensure_ascii=False, indent=4)

            print(f"Saved {label_name} and metadata to: {extracted_image_path} and {metadata_file}")

        return extracted_images

    def create_directory_structure(self):
        # Create the main directory using the docname
        base_dir = os.path.join('data/output', self.docname)

        # Create subdirectories: images, tables, texts, vectors
        subdirs = ['images', 'tables', 'texts', 'unknown']
        for subdir in subdirs:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

        # Return the base directory for further use
        return base_dir

    def normalize_bboxes(self, boxes, image_width, image_height):
        normalized_bboxes = []
        for bbox in boxes:
            x_min, y_min, x_max, y_max = bbox
            normalized_bboxes.append([
                x_min / image_width, y_min / image_height,
                x_max / image_width, y_max / image_height
            ])
        return normalized_bboxes

    def run(self):
        print(f"...Program is starting to segment {self.docname}...")

        start = timer()

        original_image = Image.open(self.path)
        image_width, image_height = original_image.size
        # shortest_edge = min(image_height, image_width)
        # longest_edge = max(image_height, image_width)

        inputs = self.processor(images=original_image, return_tensors="pt")#, size={'shortest_edge': shortest_edge, 'longest_edge': longest_edge}) #padding=True,
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([original_image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.threshold)[0]

        new_bbox_results = self.merge_bounding_boxes(results)
        final_bbox_results = self.merge_overlapping_boxes(new_bbox_results)
        normalized_boxes = self.normalize_bboxes(boxes=final_bbox_results["boxes"].tolist(), image_width=image_width, image_height=image_height)
        # self.show_labels_before(original_image.copy(), results)
        # self.show_labels_after(original_image.copy(), final_bbox_results)
        end = timer()
        print(f"Time of first section(model extract + merging of Captions and IoU boxes): {end - start}")

        start = timer()
        whitened_image = self.whiten_bounding_boxes(original_image.copy(), final_bbox_results["boxes"].tolist())
        base_dir = self.create_directory_structure()
        whitened_image_path = os.path.join(base_dir, 'images', 'whitened_document_before_merge_ocr_with_model_boxes.png')
        #whitened_image.save(whitened_image_path)
        end = timer()
        print(f"Time of second section(Whitening): {end - start}")

        start = timer()
        ocr_boxes = self.perform_ocr_easyocr(whitened_image)
        model_boxes = final_bbox_results["boxes"].tolist()
        merged_boxes, used_ocr_indices = self.merge_ocr_with_model_boxes(model_boxes, ocr_boxes, max_merges=5)
        labels = final_bbox_results["labels"].tolist()
        extracted_images = self.save_extracted_boxes(original_image.copy(), merged_boxes, labels)

        whitened_image_after_merge = self.whiten_bounding_boxes(original_image.copy(), merged_boxes)
        whitened_image_after_merge_path = os.path.join(base_dir, 'images', 'whitened_document_after_merge_ocr_with_model_boxes.png')
        whitened_image_after_merge.save(whitened_image_after_merge_path)
        end = timer()
        print(f"Time of third section(OCR and merging): {end - start}")
