#TODO: Problems to solve:
#       for PDF 2 png are created next to it.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from DocImage import DocImage
from DocumentRenderer import DocumentRenderer
import config

from pathlib import Path
from pdf2image import convert_from_path
from DocSegment import DocSegment
from DocTable import DocTable
from DocText import DocText

# def pdf_to_images(pdf_path):
#     output_dir = os.path.dirname(pdf_path)
#     images = convert_from_path(pdf_path, output_folder=output_dir, fmt="png")
#
#     image_paths = []
#     for i, image in enumerate(images):
#         img_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_page_{i + 1}.png")
#         image.save(img_path, 'PNG')
#         image_paths.append(img_path)
#     return image_paths

def pdf_to_images(pdf_path):
    # Define the output directory for the images
    output_dir = Path("../data/output/images")
    images_dir = output_dir.resolve()

    # Create the images folder if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    # Convert PDF pages to images and save them in the 'images' folder
    images = convert_from_path(pdf_path, fmt="png")

    image_paths = []
    for i, image in enumerate(images):
        img_path = images_dir / f"{Path(pdf_path).stem}_page_{i + 1}.png"
        image.save(img_path, 'PNG')
        image_paths.append(str(img_path))

    return image_paths

def main():
    path = config.path

    if path.endswith('.png'):
        docname = Path(path).stem
        segmenter = DocSegment(path=path, docname=docname)
        segmenter.run()
        tabler = DocTable(docname, multiple_in_one_detection=False, create_noise=True)
        tabler.run()

        texter = DocText(ocr_type='easyocr', language='de', docname=docname)
        texter.run()

        # doc_image = DocImage(docname, config.API_KEY)
        # doc_image.run()

        doc_renderer = DocumentRenderer(original_image_path=path, docname=docname)
        doc_renderer.run()


    elif path.endswith('.pdf'):

        images = pdf_to_images(path)

        for document_image_path in images:
            docname = Path(document_image_path).stem

            segmenter = DocSegment(path=document_image_path, docname=docname)
            segmenter.run()

            tabler = DocTable(docname, multiple_in_one_detection=False, create_noise=True)
            tabler.run()

            texter = DocText(ocr_type='easyocr', language='de', docname=docname)
            texter.run()

            # doc_image = DocImage(docname, config.API_KEY)
            # doc_image.run()

            doc_renderer = DocumentRenderer(original_image_path=document_image_path, docname=docname)
            doc_renderer.run()

    elif os.path.isdir(path):
        for file in Path(path).iterdir():
            if file.suffix.lower() in ['.pdf', '.png']:
                document_image_path = file
                docname = Path(document_image_path).stem
                if file.suffix.lower() == '.pdf':
                    images = pdf_to_images(file)
                    for document_image_path in images:
                        docname = Path(document_image_path).stem

                        segmenter = DocSegment(path=document_image_path, docname=docname)
                        segmenter.run()

                        tabler = DocTable(docname, multiple_in_one_detection=False, create_noise=True)
                        tabler.run()

                        texter = DocText(ocr_type='easyocr', language='de', docname=docname)
                        texter.run()

                        doc_image = DocImage(docname, config.API_KEY)
                        doc_image.run()

                        doc_renderer = DocumentRenderer(original_image_path=document_image_path, docname=docname)
                        doc_renderer.run()

                else:
                    segmenter = DocSegment(path=document_image_path, docname=docname)
                    segmenter.run()

                    tabler = DocTable(docname, multiple_in_one_detection=False, create_noise=True)
                    tabler.run()

                    texter = DocText(ocr_type='easyocr', language='de', docname=docname)
                    texter.run()

                    # doc_image = DocImage(docname, config.API_KEY)
                    # doc_image.run()

                    doc_renderer = DocumentRenderer(original_image_path=document_image_path, docname=docname)
                    doc_renderer.run()
    else:
        raise ValueError("Unsupported file format. Only PNG and PDF are accepted.")

if __name__ == '__main__':
    main()

