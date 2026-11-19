# Customer Feedback Insights Analyzer

**Turn customer reviews and support messages into clear, traceable business insights.**

Customer Feedback Insights Analyzer is a Streamlit application that cleans uploaded feedback, identifies common topics and sentiment, highlights urgent complaints, discovers recurring patterns, and generates weekly AI summaries with links to original customer messages.

| Capability | What users see |
|---|---|
| **CSV intake and cleaning** | Upload reviews, customer-authored support messages or a mixed CSV; view usable rows, duplicates, missing dates and excluded agent/bot replies |
| **Topic classification** | Ten topics: Delivery, Product Quality, Pricing, Refunds, Customer Support, Packaging, Website/App, Payments, Returns and Other |
| **Sentiment and urgency** | Positive, neutral or negative predictions; flags for potential safety/legal concerns |
| **Mixed feedback** | Identifies some reviews praising one aspect while criticising another, with aspect-level explanations |
| **Business dashboard** | Topic counts, sentiment distribution, weekly trends, filters and exportable review-level predictions |
| **Emerging issue detection** | Groups repeated, similar negative feedback assigned to Other or an uncertain topic; these are review candidates, not confirmed novel issues |
| **Weekly AI summaries** | Choose a week and generate complaint insights on demand with Groq; inspect supporting original review IDs and quotes |
| **Model comparison** | Switch between a TF-IDF baseline and MiniLM sentence embeddings under Advanced settings; compare held-out evaluation results |
| **Optional language handling** | Flags some Hindi/Devanagari and Hinglish feedback; optional Groq translation before English-trained classification |
| **Optional image feedback** | Upload one JPG/PNG, request an AI description, review or edit it, then analyse that confirmed description |
