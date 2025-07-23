# -------------------------------------------------------------
# FunPDFs Dockerfile
#
# USAGE INSTRUCTIONS:
#
# 1. Build the Docker image (from the FunPDFs directory):
#    docker build -t funpdfs .
#
# 2. Run the container with your PDF as input:
#    docker run --rm \
#      -v /absolute/path/to/your.pdf:/app/your.pdf \
#      funpdfs \
#      python parse_pdf_and_classify.py your.pdf tfidf_heading_classifier.joblib output.json
#
#    - Replace /absolute/path/to/your.pdf with the full path to your PDF file.
#    - The output (output.json) will be created inside the container.
#
# 3. (Optional) To save the output.json to your host machine:
#    docker run --rm \
#      -v /absolute/path/to/your.pdf:/app/your.pdf \
#      -v /absolute/path/to/output/dir:/app/output \
#      funpdfs \
#      python parse_pdf_and_classify.py your.pdf tfidf_heading_classifier.joblib output/output.json
#
#    - After running, check /absolute/path/to/output/dir/output.json on your host.
# -------------------------------------------------------------
FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script and model file
COPY parse_pdf_and_classify.py .
COPY tfidf_heading_classifier.joblib .

# Default command (can be overridden at runtime)
CMD ["python", "parse_pdf_and_classify.py"] 