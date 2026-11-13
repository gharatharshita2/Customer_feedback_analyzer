# Shipping decision — one-page note

**Recommendation:** Ship **TF-IDF (word/bigram features) + Logistic Regression** as the default small-MVP classifier **unless** the MiniLM embedding model shows a meaningful improvement on the same >=100 human-verified reviews. This is a conditional recommendation; final scores must be inserted after human labelling.

**Why the baseline first?** It starts quickly, uses modest memory, can be explained to a business owner, needs no pretrained-model download, and gives a reliable reproducible local demo. The embedding alternative captures similarities across different phrasings, but adds model download, memory and CPU costs. Use the verified per-class F1 and confusion matrices to determine whether its additional complexity is justified, particularly on infrequent complaint topics.

**What I'd ship with it:** A Streamlit dashboard with CSV validation, negative-only complaint counts, week-by-week sentiment, basic high-risk flags, uncertain-review clustering, and evidence-linked source quotes. A hosted LLM summary is part of the MVP but needed when demonstrating the assessment's **LLM** requirement; it must pass source-ID validation and fall back to extractive text on ungrounded claims. Human review is mandatory before acting on high-risk or nuanced complaints.

**Risks:** The training data is synthetic and templated; real-world language may reduce accuracy. Mixed feedback is handled using a transparent heuristic rather than full aspect-based learning. Clustering suggests possible new topics but does not prove them. LLM citations are checked against original rows, yet semantic appropriateness still deserves human review.

**Decision gate for final submission:** Fill in the actually verified topic/sentiment macro F1 and accuracy for **both** approaches. If MiniLM significantly improves performance on the same independent set, ship MiniLM instead. Never claim a winner based solely on in-sample predictions.
