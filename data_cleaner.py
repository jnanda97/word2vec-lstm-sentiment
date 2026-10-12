import re

def clean_string(s):
    s = re.sub(r"<br\s*/?>", " ", s)                      # kill the html breaks
    s = re.sub(r"[^A-Za-z0-9(),!?'`]", " ", s)            # keep letters, digits, and a few punctuation marks
    s = re.sub(r"('s|'ve|n't|'re|'d|'ll)", r" \1", s)     # split contractions
    s = re.sub(r"([,!?()])", r" \1 ", s)                  # pad punctuation
    s = re.sub(r"\s{2,}", " ", s)                         # collapse whitespace
    return s.strip().lower()
