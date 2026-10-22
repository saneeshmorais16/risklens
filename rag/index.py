"""Local TF-IDF index over the fictional policy notes."""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "rag" / "policies"


class PolicyIndex:
    def __init__(self, directory: Path = POLICY_DIR) -> None:
        self.paths = sorted(directory.glob("*.md"))
        self.texts = [path.read_text() for path in self.paths]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts) if self.texts else None

    def search(self, query: str, limit: int = 2) -> list[dict]:
        if self.matrix is None or not query.strip():
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = scores.argsort()[::-1]
        hits = []
        for index in order:
            if scores[index] <= 0.05:
                continue
            hits.append(
                {
                    "source": self.paths[index].name,
                    "excerpt": self.texts[index].strip().replace("\n", " ")[:280],
                    "score": round(float(scores[index]), 4),
                }
            )
            if len(hits) >= limit:
                break
        return hits
