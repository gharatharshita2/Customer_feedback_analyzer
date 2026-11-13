# 3–5 minute walkthrough (record personally)

**0:00–0:35 — Problem / data.** Explain that the business receives hundreds of messages; show 720 synthetic records and the cleaning metrics. Mention original-source quotes and privacy (synthetic/public only).

**0:35–1:15 — Predictions / dashboard.** Show 10 topics, 3 sentiment labels, sentiment trend, urgent flags and filters. Upload a tiny sample containing a clearly positive and a mixed review. Point to the mixed explanation: positive product quality + negative delivery.

**1:15–2:15 — Evidence-based weekly report.** Select a week in the sidebar and click **Generate LLM weekly summary** with the cloud LLM key configured. Show the validated LLM mode, negative-only counts, real evidence IDs, clickable source reviews and exact quotes. Explain unsupported claims trigger the extractive fallback. If the cloud LLM is unavailable, state openly that the extractive preview is not an LLM run.

**2:15–2:45 — New issues.** Show low-confidence / Other clustering. Explain exploratory status, source example quotes and why it can uncover loyalty/subscription issues.

**2:45–3:45 — Held-out evaluation.** Only after >=100 human-verified rows, open Model evaluation. Show the 4 results (topic/sentiment × TF-IDF/MiniLM), macro F1, accuracy, confusion matrices, and five analysed errors. Explain same held-out IDs, no evaluation text in training.

**3:45–4:30 — Decisions / ownership.** State which model you would ship and why, mention limitations of synthetic data and mixed-feedback heuristics. Show README, `src/evaluate.py`, and describe a small change you'd be comfortable making in an interview.

Upload your video to a shareable location and confirm the link permissions. A video is **not** bundled with this source-code package.


## Added support-messages intake demonstration

In the sidebar, show the single **Upload Feedback File** control. Upload `data/sample_feedback_upload.csv` (mixed reviews and support messages). Point out that customer messages and reviews are both classified, while agent/bot replies do not inflate complaint counts. Open a support-message evidence ID to show its original message, ticket ID and channel. Optionally upload `data/support_messages.csv` separately to show automatic support-export detection.
