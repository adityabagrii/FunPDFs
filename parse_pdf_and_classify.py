import fitz  # PyMuPDF
import joblib
import json
import sys
import os
import re

# Improved heuristic function for heading detection
def is_heading(line, font_size=None, is_bold=False, y_pos=None, x_pos=None, width=None, page_width=None):
    """
    Heuristic-based heading detector.
    Parameters:
    - line (str): Text line to evaluate
    - font_size (float): Optional font size (if available from PDF)
    - is_bold (bool): Optional bold indicator (if available)
    - y_pos (float): Optional vertical normalized position on page (0 = top, 1 = bottom)
    - x_pos (float): X position of the line (if available)
    - width (float): Width of the line (if available)
    - page_width (float): Width of the page (if available)
    Returns:
    - bool: True if line is likely a heading
    """
    line = line.strip()
    if not line:
        return False

    # --- Reject if clearly body text ---
    if len(line.split()) > 15:
        return False
    if '.' in line and len(line.split()) > 8:
        return False
    if line.islower():  # All lowercase
        return False
    if any(verb in line.lower().split() for verb in ['is', 'are', 'was', 'has', 'have', 'were', 'will']):
        return False
    if re.search(r'\[\d+\]', line):  # Likely a reference
        return False
    if re.match(r'^[•\-→]', line):  # Bullet points or arrows
        return False
    if re.search(r'[.;,!?]$', line) and not line.endswith(':'):
        return False

    # --- Accept if highly probable heading patterns ---
    if re.match(r'^\d+(\.\d+)*\s', line):  # e.g., "1.", "1.1", "2.3.4"
        return True
    if line.endswith(':'):
        return True
    if line.istitle():  # Title Case
        return True
    if line.isupper() and len(line) > 2:
        return True

    # --- Style-based boosts ---
    score = 0.0
    if font_size and font_size >= 12:
        score += 0.4
    if is_bold:
        score += 0.3
    if len(line.split()) < 10:
        score += 0.2
    if y_pos is not None and y_pos < 0.3:
        score += 0.1  # small boost for early-on-page
    if line.isupper() and len(line) > 2:
        score += 0.2
    # Center alignment boost
    if x_pos is not None and width is not None and page_width is not None:
        center = (x_pos + width / 2) / page_width
        if 0.45 < center < 0.55:
            score += 0.1
    # Capitalization ratio
    words = line.split()
    if words:
        cap_ratio = sum(1 for w in words if w[0].isupper()) / len(words)
        if cap_ratio > 0.7:
            score += 0.1
    # Surrounded by whitespace (not implemented here, but could be added with more context)
    return score >= 0.5

def extract_headings_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    headings = []
    for page_num, page in enumerate(doc, start=1):
        page_width = page.rect.width
        blocks = page.get_text('dict')['blocks']
        for block in blocks:
            if 'lines' not in block:
                continue
            for line in block['lines']:
                line_text = ''.join([span['text'] for span in line['spans']]).strip()
                if not line_text:
                    continue
                # Extract font size, bold, position, width
                font_sizes = [span['size'] for span in line['spans']]
                font_size = max(font_sizes) if font_sizes else None
                is_bold = any('Bold' in span.get('font', '') for span in line['spans'])
                # Use the first span for position
                x_pos = line['spans'][0]['origin'][0] if line['spans'] else None
                y_pos = line['spans'][0]['origin'][1] / page.rect.height if line['spans'] else None
                width = sum(span.get('width', 0) for span in line['spans']) if line['spans'] else None
                if is_heading(line_text, font_size, is_bold, y_pos, x_pos, width, page_width):
                    headings.append({'text': line_text, 'page': page_num})
    return headings

def main(pdf_path, model_path, output_json):
    # 1. Extract headings from PDF
    headings = extract_headings_from_pdf(pdf_path)
    if not headings:
        print('No headings detected.')
        return

    # 2. Load the trained model
    model = joblib.load(model_path)

    # 3. Classify headings
    texts = [h['text'] for h in headings]
    levels = model.predict(texts)

    # 4. Prepare output
    output = []
    for h, level in zip(headings, levels):
        output.append({
            'text': h['text'],
            'page': h['page'],
            'level': level
        })

    # 5. Write to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'Output written to {output_json}')

if _name_ == '_main_':
    if len(sys.argv) != 4:
        print('Usage: python parse_pdf_and_classify.py <input.pdf> <model.joblib> <output.json>')
        sys.exit(1)
    pdf_path = sys.argv[1]
    model_path = sys.argv[2]
    output_json = sys.argv[3]
    if not os.path.exists(pdf_path):
        print(f'PDF file not found: {pdf_path}')
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f'Model file not found: {model_path}')
        sys.exit(1)
    main(pdf_path, model_path, output_json)
