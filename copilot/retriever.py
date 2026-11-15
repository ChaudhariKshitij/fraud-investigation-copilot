"""Retrieval layer over the typology knowledge base.

Chunking: split on markdown headings, so each chunk is one idea and carries its
parent title as metadata (the title is prepended to the text for matching).

Two modes (so the improvement can be measured, not just claimed):
  baseline - word TF-IDF + cosine similarity.
  improved - fixes the failure found in evaluation, where a benign-cause chunk
             ("large legitimate purchase") outranked the suspicious-pattern chunks
             because they share words:
               1. hybrid score: word TF-IDF + character n-gram TF-IDF
               2. domain query expansion (synonyms, amounts just under 10,000)
               3. benign-chunk prior: the False Positive chunk is down-weighted
                  unless the query actually contains benign/legitimate cues
"""
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge"
BENIGN_TITLE = "False Positive Guidance"
BENIGN_CUES = re.compile(r"legitimate|genuine|benign|false positive|innocent|explain|holiday|travelling|traveling|"
                         r"salary|subscription|renewal|bill|closing|really fraud|prior history|could .* be", re.I)
EXPANSIONS = {
    r"\bhuge|enormous|big|massive|unusually large\b": "far above usual spend amount spike",
    r"\babroad|overseas|foreign|in the us\b": "unusual country never used geography",
    r"\bovernight|early morning|2am|3am\b": "night",
    r"\brapid|burst|repeated|dozen|tiny|tested|testing\b": "many small transactions within minutes velocity",
    r"\bavoid review|splitting|split|reporting limit\b": "just under reporting threshold structuring",
    r"\bconclusion|wording|word\b": "report standards risk language never certain",
}


def load_chunks(kb_dir=KB_DIR):
    chunks = []
    for f in sorted(Path(kb_dir).glob("*.md")):
        title, section, buf = "", "", []

        def flush():
            if buf and title:
                chunks.append({"source": f.name, "title": title, "section": section,
                               "text": " ".join(buf).strip()})
        for line in f.read_text().splitlines():
            if line.startswith("# "):
                flush(); title, section, buf = line[2:].strip(), "", []
            elif line.startswith("## "):
                flush(); section, buf = line[3:].strip(), []
            elif line.strip():
                buf.append(line.strip())
        flush()
    return chunks


def expand_query(q: str) -> str:
    extra = [add for pat, add in EXPANSIONS.items() if re.search(pat, q, re.I)]
    # an amount between 9,000 and 9,999 is a structuring cue
    for n in re.findall(r"\d[\d,]*", q):
        v = int(n.replace(",", "") or 0)
        if 9000 <= v <= 9999:
            extra.append("just under reporting threshold structuring")
    return q + " " + " ".join(extra)


class Retriever:
    def __init__(self, chunks=None, mode="improved"):
        assert mode in ("baseline", "improved")
        self.mode = mode
        self.chunks = chunks or load_chunks()
        docs = [f"{c['title']}. {c['section']}. {c['text']}" for c in self.chunks]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.M = self.vec.fit_transform(docs)
        if mode == "improved":
            self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.CM = self.cvec.fit_transform(docs)

    def _scores(self, query):
        if self.mode == "baseline":
            return cosine_similarity(self.vec.transform([query]), self.M).ravel()
        q = expand_query(query)
        s = 0.6 * cosine_similarity(self.vec.transform([q]), self.M).ravel() \
            + 0.4 * cosine_similarity(self.cvec.transform([q]), self.CM).ravel()
        if not BENIGN_CUES.search(query):
            for i, c in enumerate(self.chunks):
                if c["title"] == BENIGN_TITLE:
                    s[i] *= 0.5
        return s

    def search(self, query, k=3):
        sims = self._scores(query)
        top = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in top]
