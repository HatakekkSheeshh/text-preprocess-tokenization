import re
import html
from pathlib import Path

DATASET_NAME = "enwik8"

PROJECT_ROOT = Path.cwd().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / DATASET_NAME / DATASET_NAME     # Sửa chỗ này nếu lỗi
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / DATASET_NAME              # Sửa chỗ này
CLEAN_DATA_PATH = PROCESSED_DIR / f"{DATASET_NAME}.txt"                         # Rename chỗ này cho khớp với dataloader

TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

WIKI_PARAM_PATTERN = re.compile(r"\{\{\{[^{}]*\}\}\}") 
WIKI_TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]*\}\}")
WIKI_TABLE_PATTERN = re.compile(r"\{\|.*?\|\}", flags=re.DOTALL)

MATH_PATTERN = re.compile(r"<math.*?>.*?</math>", flags=re.DOTALL | re.IGNORECASE)

WIKI_FILE_PATTERN = re.compile(r"\[\[(?:image|category|file):(?:\[\[.*?\]\]|[^\]]|\](?!\]))*\]\]", flags=re.DOTALL | re.IGNORECASE)
WIKI_LINK_PATTERN = re.compile(r"\[\[(?:[^|\]]*\|)*([^|\]]+)\]\]")
WIKI_FORMAT_PATTERN = re.compile(r"^[*\#;:]+\s*|''+", flags=re.MULTILINE)
WS_PATTERN = re.compile(r"\s+")
REDIRECT_PATTERN = re.compile(r"^(?:#redirect|redirect\s*\]).*$", flags=re.IGNORECASE)

HEADING_PATTERN = re.compile(r"={2,}.*?={2,}")
BRACKET_PATTERN = re.compile(r"\[.*?\]", flags=re.DOTALL)
EMPTY_PAREN_PATTERN = re.compile(r"\(\s*(?:ipa|and:\s*)?[^a-z0-9]*\s*\)", flags=re.IGNORECASE)
ENTITY_PATTERN = re.compile(r"&[a-z]+;|&#\d+;")

GARBAGE_TAIL_PATTERN1 = re.compile(
    r"==\s*(?:see also|references|external links|notes|bibliography|further reading|overview)\s*==.*", 
    flags=re.IGNORECASE | re.DOTALL
)

COMMENT_PATTERN = re.compile(r"", flags=re.DOTALL)
ALPHABET_INDEX_PATTERN = re.compile(r"(?:[a-z]\s*[-|]\s*){4,}")
LIST_PATTERN = re.compile(r"^[*\#;:]+\s*", flags=re.MULTILINE)
BOLD_ITALIC_PATTERN = re.compile(r"''+") 
PAREN_START_PUNCT_PATTERN = re.compile(r"\(\s*[,;:/]+\s*")
PAREN_END_PUNCT_PATTERN = re.compile(r"\s*[,;:/]+\s*\)")

def clean_text(text):
    text = html.unescape(text)
    text = html.unescape(text)
    text = text.lower()

    text = MATH_PATTERN.sub(" ", text)
    
    text = GARBAGE_TAIL_PATTERN1.sub(" ", text)

    while "{{" in text or "{|" in text or "{{{" in text:
        prev_text = text
        text = WIKI_PARAM_PATTERN.sub(" ", text) 
        text = WIKI_TEMPLATE_PATTERN.sub(" ", text)
        text = WIKI_TABLE_PATTERN.sub(" ", text)
        if text == prev_text:
            break

    text = TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = HEADING_PATTERN.sub(" ", text)

    text = WIKI_FILE_PATTERN.sub(" ", text)
    text = WIKI_LINK_PATTERN.sub(r"\1", text)
    text = BRACKET_PATTERN.sub(" ", text)
    text = ENTITY_PATTERN.sub(" ", text)

    text = EMPTY_PAREN_PATTERN.sub(" ", text)
    text = EMPTY_PAREN_PATTERN.sub(" ", text)

    text = PAREN_START_PUNCT_PATTERN.sub("(", text)
    text = PAREN_END_PUNCT_PATTERN.sub(")", text)

    text = LIST_PATTERN.sub(" ", text)
    text = BOLD_ITALIC_PATTERN.sub("", text)
    text = REDIRECT_PATTERN.sub(" ", text)
    text = WS_PATTERN.sub(" ", text)
    
    return text.strip()

def extract_and_clean_enwik8():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {RAW_DATA_PATH}")
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    is_inside_text = False
    article_buffer = []
    
    with RAW_DATA_PATH.open("r", encoding="utf-8", errors="ignore") as f_in, \
         CLEAN_DATA_PATH.open("w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if "<text" in line:
                is_inside_text = True
                line = line.split(">", 1)[-1]
                
            if not is_inside_text:
                continue
                
            if "</text>" in line:
                line = line.split("</text>")[0]
                article_buffer.append(line)
                is_inside_text = False 
            else:
                article_buffer.append(line)
                
            if not is_inside_text and article_buffer:
                full_article = "".join(article_buffer)
                cleaned_article = clean_text(full_article)
                
                for c_line in cleaned_article.split("\n"):
                    c_line = c_line.strip()
                    
                    if "redirect" in c_line:
                        continue
                    
                    if c_line.startswith("==") and c_line.endswith("=="):
                        continue   
                    if c_line.startswith(("image:", "category:", "file:", "[", "|", "!", "!--", "{{")):
                        continue         
                    if re.match(r"^[a-z\-]{2,15}:", c_line):
                        continue
                    if c_line.startswith(("list of", "lists of", "see also", "by iata", "by icao")):
                        continue
                    if "see :category:" in c_line or "see also:" in c_line:
                        continue
                        
                    if ALPHABET_INDEX_PATTERN.search(c_line):
                        continue
                    
                    if len(c_line) > 10:
                        f_out.write(c_line + "\n")                  # Ở đây thay bằng gì cx đc eos, \n, ...
                        
                article_buffer = []