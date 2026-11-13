"""Transparent, conservative clause-level explanation of mixed feedback.

The model remains single-label for fair scoring; these rules are a small
post-processing layer used to avoid hiding an explicit negative aspect.
"""
from __future__ import annotations
import re

TOPIC_HINTS = {
    "Delivery": r"\b(?:deliver(?:y|ed)?|shipping|shipment|arriv(?:e|ed|al)|courier|tracking|parcel)\b",
    "Product Quality": r"\b(?:product|item|quality|material|battery|device|fabric|durab(?:le|ility))\b",
    "Pricing": r"\b(?:price|pricing|expensive|cheap|cost|discount|overpriced)\b",
    "Refunds": r"\b(?:refund|reimbursement|money back)\b",
    "Customer Support": r"\b(?:support|agent|representative|service team|helpdesk)\b",
    "Packaging": r"\b(?:packag(?:ing|e)|box|wrapp(?:ing|ed))\b",
    "Website/App": r"\b(?:website|app|checkout|login|browser)\b",
    "Payments": r"\b(?:payment|charged|transaction|debit|credit card)\b",
    "Returns": r"\b(?:return|exchange|send back)\b",
}
NEGATIVE = r"\b(?:late|delayed|broken|bad|poor|terrible|awful|damaged|defective|failed|faulty|worst|missing|slow|expensive|overpriced|incorrect|unreliable|rude|frustrating|never arrived|doesn.t work|double.charged)\b"
POSITIVE = r"\b(?:great|excellent|good|fantastic|fast|quick|on time|easy|helpful|amazing|perfect|reliable|smooth|love|loved|satisfied|works well|arrived early)\b"
NEGATIONS = r"\b(?:not|never|no|isn.t|wasn.t|didn.t|don.t)\b"

def analyse_aspects(text: str) -> dict:
    clauses = re.split(r"\s*(?:[.;!]\s*|\bbut\b|\bhowever\b|\balthough\b|\byet\b)\s*", str(text), flags=re.I)
    aspects = []
    all_positive, all_negative = False, False
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        is_negative = bool(re.search(NEGATIVE, clause, re.I))
        is_positive = bool(re.search(POSITIVE, clause, re.I))
        # A simple negation heuristic; don't claim full linguistic understanding.
        if re.search(r"\b(?:not|never)\s+(?:good|great|helpful|reliable|working|fast)\b", clause, re.I):
            is_negative, is_positive = True, False
        elif re.search(r"\b(?:not|never)\s+(?:late|bad|broken|slow|damaged)\b", clause, re.I):
            is_negative, is_positive = False, True
        all_positive = all_positive or is_positive
        all_negative = all_negative or is_negative
        if is_negative == is_positive:
            continue
        sentiment = "negative" if is_negative else "positive"
        topics = [topic for topic, pattern in TOPIC_HINTS.items() if re.search(pattern, clause, re.I)]
        for topic in topics[:2]:
            aspects.append({"topic": topic, "sentiment": sentiment, "clause": clause})
    negative = [a for a in aspects if a["sentiment"] == "negative"]
    positive = [a for a in aspects if a["sentiment"] == "positive"]
    return {"aspects": aspects, "mixed": bool(negative and positive), "negative_aspect": negative[0] if negative else None, "positive_aspect": positive[0] if positive else None, "all_positive": all_positive, "all_negative": all_negative}

def explain_review(text: str, predicted_topic: str, predicted_sentiment: str) -> dict:
    analysis = analyse_aspects(text)
    topic, sentiment, explanation = predicted_topic, predicted_sentiment, ""
    if analysis["mixed"]:
        bad = analysis["negative_aspect"]
        good = analysis["positive_aspect"]
        topic, sentiment = bad["topic"], "negative"
        explanation = f"Mixed feedback: positive {good['topic']} ('{good['clause']}'); negative {bad['topic']} ('{bad['clause']}'). For complaint tracking, the negative aspect takes priority."
    elif analysis["all_positive"] and not analysis["all_negative"]:
        # Only override a clearly identified positive aspect; avoids a plainly
        # positive review being counted as an unhappy customer's complaint.
        sentiment = "positive"
    return {"topic": topic, "sentiment": sentiment, "mixed": analysis["mixed"], "explanation": explanation}
