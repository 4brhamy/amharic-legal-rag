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
!apt-get -qq install -y tesseract-ocr poppler-utils > /dev/null
!wget -q -O /usr/share/tesseract-ocr/4.00/tessdata/amh.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/amh.traineddata 2>/dev/null || wget -q -O /usr/share/tesseract-ocr/5/tessdata/amh.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/amh.traineddata
!tesseract --version | head -1 && tesseract --list-langs | grep amh
!rm -rf amharic-legal-rag && git clone -q https://github.com/4brhamy/amharic-legal-rag.git
import sys
sys.path.insert(0, "amharic-legal-rag")

# %%
import os
if not os.path.exists("family_code_am.pdf"):
    from google.colab import files
    up = files.upload()
    os.rename(next(iter(up)), "family_code_am.pdf")
!wget -q -O family_code_en.pdf https://faolex.fao.org/docs/pdf/eth208224.pdf
!pdfinfo family_code_am.pdf | grep Pages && pdfinfo family_code_en.pdf | grep Pages

# %% [markdown]
# ## OCR every Amharic page
# Pages are rendered at 300 DPI in greyscale. `--psm 4` treats each page as a single column of text of
# variable size, which matches the layout (headings, numbered sub-articles, indented lines).

# %%
import glob, subprocess
from concurrent.futures import ThreadPoolExecutor

os.makedirs("pages", exist_ok=True)
os.makedirs("ocr", exist_ok=True)
!pdftoppm -r 300 -gray -png family_code_am.pdf pages/p

def ocr(png):
    out = os.path.join("ocr", os.path.basename(png)[:-4])
    subprocess.run(["tesseract", png, out, "-l", "amh", "--psm", "4"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out + ".txt"

pngs = sorted(glob.glob("pages/*.png"))
with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
    txts = list(pool.map(ocr, pngs))
pages_am = [open(t, encoding="utf-8").read() for t in txts]
print(len(pages_am), "pages OCR'd")
print(pages_am[9][:800])

# %% [markdown]
# ## Split into articles and check the result
# The law has 323 articles. OCR occasionally misreads an Ethiopic numeral; the splitter checks every
# heading against the running sequence and flags any number it had to infer.

# %%
from amlegal.corpus import clean_page, split_articles, split_english, save_jsonl

arts_am = split_articles([clean_page(p) for p in pages_am])
nums = [a.number for a in arts_am]
missing = [n for n in range(1, 324) if n not in nums]
unchecked = [a.number for a in arts_am if not a.number_checked]
print(f"Amharic articles: {len(arts_am)} (last = {nums[-1] if nums else None})")
print(f"Missing numbers ({len(missing)}): {missing}")
print(f"Numbers inferred from the sequence ({len(unchecked)}): {unchecked}")

!pdftotext -layout family_code_en.pdf family_code_en.txt
arts_en = split_english(open("family_code_en.txt", encoding="utf-8").read())
print(f"English articles: {len(arts_en)}")

# %% [markdown]
# ## Merge into one bilingual record per article

# %%
en = {a.number: a for a in arts_en}
am = {a.number: a for a in arts_am}
records = []
for n in range(1, 324):
    a, e = am.get(n), en.get(n)
    records.append({
        "article": n,
        "title_am": a.title if a else "",
        "text_am": a.text if a else "",
        "title_en": e.title if e else "",
        "text_en": e.text if e else "",
        "page_am": a.page if a else None,
        "number_checked": a.number_checked if a else False,
    })

os.makedirs("out", exist_ok=True)
save_jsonl(records, "out/family_code.jsonl")
both = sum(1 for r in records if r["text_am"] and r["text_en"])
print(f"{both}/323 articles have both Amharic and English text")

import random
for r in random.sample(records, 3):
    print(f"\n--- Article {r['article']} ---\n{r['title_am']}\n{r['text_am'][:300]}\n\n{r['title_en']}\n{r['text_en'][:300]}")

# %% [markdown]
# ## Save the outputs
# `out/family_code.jsonl` is the corpus used by the rest of the project. The raw OCR pages are kept
# too, so the OCR can be inspected or corrected later.

# %%
!cp -r ocr out/ocr_pages && cd out && zip -qr ../corpus_outputs.zip . && cd .. && ls -la corpus_outputs.zip
