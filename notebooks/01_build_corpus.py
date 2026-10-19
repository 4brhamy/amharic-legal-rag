# %% [markdown]
# # 1. Build the bilingual Family Code corpus
#
# Turns the scanned Amharic Revised Family Code (Proclamation No. 213/2000 · አዋጅ ቁጥር 213/1992) into one
# record per article, matched with the English text of the same article.
#
# - **Amharic:** the scan has no usable text layer, so every page is OCR'd with Tesseract's `amh` model
#   (`tessdata_best`).
# - **English:** the FAOLEX copy has a clean text layer, extracted with `pdftotext`.
#
# **Runtime:** CPU is enough (about 5-10 minutes for 99 pages). Before running, upload the Amharic PDF
# as `family_code_am.pdf` in the Colab file panel, or run the cell below and pick the file.

# %%
!apt-get -qq update > /dev/null && apt-get -qq install -y tesseract-ocr poppler-utils > /dev/null
!pip -q install opencv-python-headless
# Most accurate Amharic model, kept in a local folder and passed with --tessdata-dir
!mkdir -p tessdata && wget -q -O tessdata/amh.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/amh.traineddata
!tesseract --version | head -1 && tesseract --tessdata-dir tessdata --list-langs | grep amh && pdftoppm -v 2>&1 | head -1
!rm -rf amharic-legal-rag && git clone -q https://github.com/4brhamy/amharic-legal-rag.git
import sys
sys.path.insert(0, "amharic-legal-rag")

# %%
import json, os
if not os.path.exists("family_code_am.pdf"):
    from google.colab import files
    up = files.upload()
    os.rename(next(iter(up)), "family_code_am.pdf")
!wget -q -O family_code_en.pdf https://faolex.fao.org/docs/pdf/eth208224.pdf
!pdfinfo family_code_am.pdf | grep Pages && pdfinfo family_code_en.pdf | grep Pages

# %% [markdown]
# ## OCR every Amharic page (two passes)
# Pages are rendered at 300 DPI. What works for this scan was found by testing on problem pages:
# - **Underline removal.** Headings are underlined, which makes Tesseract misread them.
# - **Two layout modes.** `--psm 4` and `--psm 6` each drop *different* underlined headings (for
#   example psm 4 loses "Article 112 – Appeal" on page 41, psm 6 finds it). Both passes are kept and
#   the build script picks, page by page, the pass that found more headings.
#
# Each pass takes roughly 15 minutes on Colab's CPU.

# %%
import glob
from concurrent.futures import ThreadPoolExecutor
from amlegal.ocr import ocr_image

RAW = "amharic-legal-rag/data/raw"
!pdftoppm -r 300 -gray -png family_code_am.pdf pages/p 2>/dev/null || (mkdir -p pages && pdftoppm -r 300 -gray -png family_code_am.pdf pages/p)
pngs = sorted(glob.glob("pages/*.png"))
for psm in (4, 6):
    out_dir = f"{RAW}/ocr_psm{psm}"
    os.makedirs(out_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        texts = list(pool.map(lambda p: ocr_image(p, psm=psm, tessdata_dir="tessdata"), pngs))
    for png, text in zip(pngs, texts):
        open(f"{out_dir}/{os.path.basename(png)[:-4]}.txt", "w", encoding="utf-8").write(text)
    print(f"psm {psm}: {len(texts)} pages")

# %% [markdown]
# ## English articles
# The English text has a clean text layer; it is split on its "Article N." headings (323 articles).

# %%
from amlegal.corpus import split_english
!pdftotext -layout family_code_en.pdf family_code_en.txt
arts_en = split_english(open("family_code_en.txt", encoding="utf-8").read())
print(f"English articles: {len(arts_en)}")
with open(f"{RAW}/family_code.jsonl", "w", encoding="utf-8") as f:
    for a in arts_en:
        f.write(json.dumps({"article": a.number, "title_en": a.title, "text_en": a.text}, ensure_ascii=False) + "\n")

# %% [markdown]
# ## Align Amharic to English and build the corpus
# OCR misreads most Ethiopic article numerals, so numbers are not read from the headings. Instead each
# Amharic block is aligned to the English articles by length, number of sub-articles and any numeral
# that did read correctly (`amlegal/align.py`). Every record carries an `alignment` flag:
# `1:1` is a clean match; `1:2`/`1:3` means OCR missed a heading and neighbouring articles share a block.

# %%
!cd amharic-legal-rag && python scripts/build_corpus.py

# %% [markdown]
# ## Save the outputs

# %%
!cd amharic-legal-rag && zip -qr ../corpus_outputs.zip data && cd .. && ls -la corpus_outputs.zip
from google.colab import files
files.download("corpus_outputs.zip")
