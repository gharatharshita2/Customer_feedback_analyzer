# StarvoxLabs submission checklist

The following items cannot legitimately be completed on the candidate's behalf. Complete them before the five-calendar-day deadline measured from acknowledging the assignment.

- [x] Source code: Streamlit dashboard, one CSV uploader accepting reviews, customer-authored support messages or a mixed file, role-aware message filtering, source/ticket traceability, CSV cleaning, topic/sentiment models, urgent flags, uncertain-topic grouping and quote evidence.
- [x] Two approaches implemented: classic TF-IDF + logistic regression; MiniLM sentence embeddings + logistic regression.
- [x] Hosted Groq LLM summarization implemented with citation/claim checking and extractive fallback; **live API call still needs to be demonstrated**.
- [x] 720 synthetic reviews and 120 independent AI-drafted label candidates.
- [x] Python unit tests created; pass without fetching external models.
- [ ] **Human-read and verify** at least 100 evaluation rows; mark `human_verified=yes` only once checked.
- [ ] Run **both** approaches on the verified data; save `outputs/evaluation_results.json`, results table, all confusion matrices and prediction files.
- [ ] Explain **at least five actual incorrect predictions** after validation. Update `outputs/mistakes_to_analyse.csv` / a short error analysis file.
- [ ] Configure GROQ_API_KEY in Streamlit secrets and run a **real hosted LLM weekly summary**; check citations and add the demo to the walkthrough.
- [ ] Add verified accuracy and F1 to the one-page shipping decision note; choose model based on results.
- [ ] Push the project to a GitHub repository accessible to reviewers.
- [ ] Record and share a 3–5-minute walkthrough showing your actual working application.
- [ ] Submit a live demo URL **or** clear local instructions (see `docs/DEPLOYMENT.md` and the README). If hosting, configure the private API key and verify one actual LLM-generated summary.
- [ ] Truthfully state your own work, what AI assisted with, and be ready to explain trade-offs and make a live code change.
- [ ] Reply to the recruitment email within five calendar days of acknowledging the task, with the requested links.
