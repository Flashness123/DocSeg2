# DocSeg

Document layout analysis pipeline I built for my bachelor thesis at intergator
in Dresden. It takes a scanned page (PDF or PNG) and turns it into
structured regions — text blocks, tables, figures — with OCR'd text and rendered
overlays.

The documents I worked on were German technical drawings and engineering specs,
so the defaults are tuned for German (`easyocr`, `language='de'`).

## Pipeline

Each page runs through four stages, orchestrated in `src/main.py`:

1. **Layout segmentation** (`DocSegment.py`) — `Aryn/deformable-detr-DocLayNet`
   (Deformable DETR) detects regions and labels them (text, table, figure,
   title, ...). Includes some post-processing to merge boxes by proximity and
   drop overlaps by IoU.
2. **Tables** (`DocTable.py`) — `microsoft/table-transformer-detection` finds
   tables, `table-transformer-structure-recognition` recovers rows/columns.
3. **Text** (`DocText.py`) — OCR per region, either `easyocr` or `pytesseract`.
4. **Rendering** (`DocumentRenderer.py`) — draws the detected regions and labels
   back onto the page.

There's also an optional `DocImage.py` stage that captions figures with the
OpenAI API; it's commented out in `main.py` by default since it needs a key.

## Running it

Dependencies are pinned in `build/` (`requirements.txt`, and a fully resolved
`complete.txt`). Setup steps are in `build/setup-instructions.sh` — roughly:

```bash
sudo apt install tesseract-ocr        # only if you use the tesseract OCR path
pip install -r build/requirements.txt
```

Point `config.py` at the input you want (`path`) and run:

```bash
cd src
python main.py
```

Outputs land under `data/output/<docname>/`. The `data/` folder here contains
the real thesis test documents and their generated output so you can see what
the pipeline produces.

Config lives in `src/config.py`: `path` (input file or dir), `TESSERACT_PATH`
(Windows tesseract binary), `API_KEY` (for the optional OpenAI captioning), and
`FONT_PATH`.

## Notes

This is thesis code, not a packaged library — expect rough edges. The write-up
is in `doc/DocSeg.pdf`.
