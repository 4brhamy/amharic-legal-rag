"""OCR for scanned Amharic legal documents (Tesseract + OpenCV preprocessing).

Headings in the Negarit Gazeta scans are underlined. Tesseract misreads underlined text: Ethiopic
numerals come out wrong and some headings are dropped entirely. Erasing long horizontal lines before
OCR fixes most of this.
"""
import subprocess
import tempfile

import cv2
import numpy as np


def remove_underlines(gray: np.ndarray, min_line_frac: float = 0.04) -> np.ndarray:
    """Return a black-on-white binary image with long horizontal lines erased.

    A line counts as an underline if it is at least `min_line_frac` of the page width long and only
    a few pixels tall, which letters never are.
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # text = white
    width = max(10, int(bw.shape[1] * min_line_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    cleaned = cv2.subtract(bw, lines)
    # close the small gaps that erasing a line can leave in letters sitting on it
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return 255 - cleaned


def ocr_image(path: str, lang: str = "amh", psm: int = 4, tessdata_dir: str | None = None,
              preprocess: bool = True) -> str:
    """OCR one page image and return its text."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if preprocess:
        img = remove_underlines(img)
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        cv2.imwrite(tmp.name, img)
        cmd = ["tesseract", tmp.name, "stdout", "-l", lang, "--psm", str(psm)]
        if tessdata_dir:
            cmd[1:1] = ["--tessdata-dir", tessdata_dir]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout
