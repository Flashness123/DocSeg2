sudo apt install tesseract-ocr

pyenv virtualenv <VERSION> doc-sec-env
pyenv shell doc-sec-env 
python -m pip install --upgrade pip
pip install pip-tools
pip-compile --resolver=backtracking --output-file=complete.txt requirements.txt
pip install --no-cache-dir --no-deps -r complete.txt
