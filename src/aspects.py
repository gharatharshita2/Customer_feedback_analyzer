"""Conservative aspect explanations (not part of the held-out ML score).

Reviews retain ONE primary label for the assignment. Explicitly described
positive and negative aspects are additionally exposed for inspection.
"""
from __future__ import annotations
import re

TOPIC_HINTS = {
    "Delivery": r"\b(?:deliver\w*|shipp\w*|shipment|arriv\w*|courier|tracking|parcel)\b",
    "Product Quality": r"\b(?:product|item|quality|material|battery|device|fabric|durab\w*|charger|headphones?|speaker|stitching|motor|blender|kettle|shoes?)\b",
    "Pricing": r"\b(?:price|pricing|expensive|cheap|cost|discount|overpriced|value)\b",
    "Refunds": r"\b(?:refund\w*|reimbursement|money back)\b",
    "Customer Support": r"\b(?:support|agent|representative|service team|helpdesk|customer care)\b",
    "Packaging": r"\b(?:packag\w*|box|wrapp\w*|packing|sealed)\b",
    "Website/App": r"\b(?:website|app|checkout|login|browser|cart|search filter)\b",
    "Payments": r"\b(?:payment|charged|transaction|debit|credit card|upi)\b",
    "Returns": r"\b(?:return\w*|exchange|send back|pickup)\b",
}
NEGATIVE = r"\b(?:late|delay\w*|broken|bad|poor|terrible|awful|damag\w*|defect\w*|fail\w*|faulty|worst|missing|slow|expensive|overpriced|incorrect|unreliable|rude|frustrating|lost|crack\w*|leak\w*|unresponsive|problem\w*|issue\w*|stopped working|not working|never arrived|doesn.t work|double.charged)\b"
POSITIVE = r"\b(?:great|excellent|good|fantastic|fast|quick|on time|easy|helpful|amazing|perfect|reliable|smooth|love|loved|satisfied|works well|arrived early|lovely|premium)\b"


def _sentiment(clause: str) -> str | None:
    lower = clause.casefold()
    if re.search(r"\b(?:not|never|isn.t|wasn.t|didn.t|don.t)\s+(?:good|great|helpful|reliable|working|fast|happy|on time)\b", lower):
        return "negative"
    if re.search(r"\b(?:not|never|isn.t|wasn.t|didn.t)\s+(?:late|bad|broken|slow|damaged|delayed)\b", lower):
        return "positive"
    neg = bool(re.search(NEGATIVE, lower))
    pos = bool(re.search(POSITIVE, lower))
    if neg == pos:
        return None  # uncertain, do not invent an aspect
    return "negative" if neg else "positive"


def analyse_aspects(text: str) -> dict:
    clauses = re.split(r"\s*(?:[.;!]\s*|\bbut\b|\bhowever\b|\balthough\b|\byet\b)\s*", str(text), flags=re.I)
    aspects = []
    positive_seen = negative_seen = False
    for clause in clauses:
        clause = clause.strip(' ,')
        if not clause:
            continue
        sentiment = _sentiment(clause)
        if sentiment is None:
            continue
        positive_seen |= sentiment == "positive"
        negative_seen |= sentiment == "negative"
        topics = [topic for topic, pattern in TOPIC_HINTS.items() if re.search(pattern, clause, re.I)]
        # 'The product arrived damaged' is quality feedback; arrival alone isn't a shipping complaint.
        if "Product Quality" in topics and "Delivery" in topics and sentiment == "negative":
            if re.search(r"\b(?:damag\w*|broken|defect\w*|faulty|crack\w*)\b", clause, re.I) and not re.search(r"\b(?:late|delay\w*|courier|shipp\w*|tracking)\b", clause, re.I):
                topics.remove("Delivery")
        for topic in topics:
            aspects.append({"topic": topic, "sentiment": sentiment, "clause": clause})
    negative = [a for a in aspects if a['sentiment'] == 'negative']
    positive = [a for a in aspects if a['sentiment'] == 'positive']
    return {"aspects": aspects, "mixed": bool(positive and negative),
            "negative_aspect": negative[0] if negative else None,
            "positive_aspect": positive[0] if positive else None,
            "all_positive": positive_seen, "all_negative": negative_seen}


def explain_review(text: str, predicted_topic: str, predicted_sentiment: str) -> dict:
    analysis = analyse_aspects(text)
    topic, sentiment, explanation = predicted_topic, predicted_sentiment, ""
    if analysis['mixed']:
        bad, good = analysis['negative_aspect'], analysis['positive_aspect']
        topic, sentiment = bad['topic'], 'negative'
        explanation = (f"Mixed feedback: positive {good['topic']} ('{good['clause']}'); "
                       f"negative {bad['topic']} ('{bad['clause']}'). "
                       "Both aspects are shown; the complaint is the primary label.")
    elif analysis['all_positive'] and not analysis['all_negative']:
        sentiment = 'positive'
    return {"topic": topic, "sentiment": sentiment, "mixed": analysis['mixed'],
            "explanation": explanation, "aspects": analysis['aspects']}
