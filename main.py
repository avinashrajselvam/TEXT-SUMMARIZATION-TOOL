"""
Text Summarization Tool - CODTECH Deliverable
- Two summarization methods:
  1) frequency_based_summary (pure-Python, no external deps)
  2) tfidf_summary (uses scikit-learn if available)
- The script includes example long texts and prints concise summaries.
- Save this file and run: python text_summarizer.py
"""

import re
import math
from collections import Counter, defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

def clean_text(text):
    # basic cleaning: remove newlines, multiple spaces, keep simple punctuation for sentence splitting
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def split_sentences(text):
    # naive sentence splitter based on punctuation. Good enough for demo samples.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def tokenize(text):
    # lowercase, remove non-alphanumeric except spaces
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9']+\b", text)
    return tokens

STOPWORDS = {
    # small stopword set for demonstration
    'the','and','is','in','it','of','to','a','that','as','for','with','on','are','was','by','an','be','this','which','or','from','at','has','have'
}

def frequency_based_summary(text, max_sentences=3):
    text = clean_text(text)
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return ' '.join(sentences)
    words = tokenize(text)
    # compute word frequencies excluding stopwords and short words
    freqs = Counter(w for w in words if w not in STOPWORDS and len(w) > 2)
    if not freqs:
        # fallback: return first N sentences
        return ' '.join(sentences[:max_sentences])
    # normalize frequencies
    maxf = max(freqs.values())
    for k in freqs:
        freqs[k] /= maxf
    # score sentences by sum of normalized word frequencies
    sent_scores = []
    for s in sentences:
        s_words = tokenize(s)
        score = sum(freqs.get(w, 0) for w in s_words)
        # normalize by sentence length to avoid bias to long sentences
        if s_words:
            score /= math.sqrt(len(s_words))
        sent_scores.append((score, s))
    # pick top sentences by score, preserve original order
    top = sorted(sent_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_set = set(t[1] for t in top)
    summary_sentences = [s for s in sentences if s in top_set]
    return ' '.join(summary_sentences)

def tfidf_summary(text, max_sentences=3):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not available in environment. Cannot run tfidf_summary.")
    text = clean_text(text)
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return ' '.join(sentences)
    # vectorize sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    # score sentences by the sum of TF-IDF values (words with high tfidf indicate importance)
    scores = X.sum(axis=1).A1  # convert to 1d array
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:max_sentences]
    top_idx = sorted(i for i, _ in ranked)
    summary_sentences = [sentences[i] for i in top_idx]
    return ' '.join(summary_sentences)

def demo():
    long_text = (
        "Artificial intelligence (AI) is transforming industries across the world. "
        "From healthcare to finance, AI systems assist experts by identifying patterns in data "
        "and suggesting optimal actions. Machine learning, a subset of AI, uses statistical methods "
        "to enable machines to improve with experience. Deep learning further builds on machine learning "
        "with multi-layered neural networks that can learn hierarchical representations. "
        "However, AI adoption also raises important ethical questions. Issues such as data privacy, "
        "bias in training datasets, and job displacement require thoughtful policy and engineering choices. "
        "Governments and organizations are increasingly investing in AI research to drive innovation while "
        "creating frameworks for responsible use. Education is vital: a workforce literate in AI concepts "
        "will be better prepared to collaborate with intelligent systems. In the years ahead, AI's role "
        "will likely expand, reshaping how we work, learn, and solve complex problems."
    )
    long_text_2 = (
        "Climate change is one of the most pressing challenges of our time. Rising global temperatures "
        "are linked to increased greenhouse gas emissions from human activities like fossil fuel burning "
        "and deforestation. The consequences are widespread: more intense storms, sea level rise, "
        "shifts in agricultural productivity, and threats to biodiversity. Mitigation strategies include "
        "reducing emissions through renewable energy, enhancing energy efficiency, and protecting forests. "
        "Adaptation measures—such as building resilient infrastructure and developing drought-resistant crops—"
        "are also necessary to reduce vulnerability. International cooperation, technology transfer, and "
        "climate finance play key roles in enabling countries to respond. Rapid action this decade is critical "
        "to limit warming and avoid the most dangerous impacts."
    )
    samples = [("AI Article", long_text), ("Climate Article", long_text_2)]
    print("=== CODTECH TEXT SUMMARIZER DEMO ===\\n")
    for title, txt in samples:
        print(f"--- {title} (input length: {len(txt)} chars) ---\\n")
        print("Original (first 300 chars):")
        print(txt[:300] + ("..." if len(txt) > 300 else ""))
        print("\\n[Frequency-based summary]")
        print(frequency_based_summary(txt, max_sentences=2))
        if SKLEARN_AVAILABLE:
            print("\\n[TF-IDF summary]")
            print(tfidf_summary(txt, max_sentences=2))
        else:
            print("\\n[TF-IDF summary] -- SKLEARN NOT AVAILABLE; skipping.\\n")
        print("-" * 60 + "\\n")

if __name__ == '__main__':
    demo()
