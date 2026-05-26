import pdfplumber
import docx
import os

def parse_resume(file_path):
    """
    Parses PDF or DOCX resume and returns the text content.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return parse_pdf(file_path)
    elif ext == '.docx':
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def parse_pdf(file_path):
    """
    FIX 5: Multi-column PDF parsing using bounding boxes.
    Instead of naive horizontal text extraction, we extract each word with its
    bounding box position and re-sort them in reading order (top-to-bottom,
    then left-to-right within detected column boundaries). This correctly
    handles modern multi-column resume layouts.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_width = page.width

                # Extract words with their bounding boxes
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False
                )

                if not words:
                    text += page.extract_text() or ""
                    continue

                # Detect if multi-column: check if words span a wide x range
                x_positions = [w['x0'] for w in words]
                x_min, x_max = min(x_positions), max(x_positions)
                is_multi_column = (x_max - x_min) > (page_width * 0.45)

                if is_multi_column:
                    # Determine column midpoint
                    mid_x = page_width / 2.0

                    # Split words into left and right columns
                    left_words = [w for w in words if w['x0'] < mid_x]
                    right_words = [w for w in words if w['x0'] >= mid_x]

                    # Sort each column top-to-bottom
                    left_words = sorted(left_words, key=lambda w: (round(w['top'] / 5), w['x0']))
                    right_words = sorted(right_words, key=lambda w: (round(w['top'] / 5), w['x0']))

                    # Reconstruct text: left column first, then right column
                    def words_to_text(word_list):
                        lines = {}
                        for w in word_list:
                            line_key = round(w['top'] / 5)
                            lines.setdefault(line_key, []).append(w['text'])
                        return "\n".join(" ".join(line) for line in sorted(lines.values(), key=lambda l: l))

                    text += words_to_text(left_words) + "\n" + words_to_text(right_words) + "\n"
                else:
                    # Single column: use standard extraction
                    text += page.extract_text() or ""

    except Exception as e:
        print(f"Error parsing PDF: {e}")
        # Fallback: naive full-page extraction
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e2:
            print(f"Fallback parsing also failed: {e2}")
    return text

def parse_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error parsing DOCX: {e}")
        return ""
