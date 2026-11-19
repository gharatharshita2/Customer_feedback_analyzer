# Shipping Decision – Customer Feedback Insights Analyzer

## Recommended Approach

I recommend **MiniLM Sentence Embeddings with Logistic Regression** as the default classification approach for the Customer Feedback Insights Analyzer.

Both MiniLM and TF-IDF were implemented and evaluated on the same 100 human-reviewed customer feedback examples.

## Evaluation Results

| Task | TF-IDF Accuracy | MiniLM Accuracy |
|---|---:|---:|
| Topic Classification | 51% | 85% |
| Sentiment Analysis | 63% | 74% |

| Task | TF-IDF Macro F1 | MiniLM Macro F1 |
|---|---:|---:|
| Topic Classification | 0.4623 | 0.7522 |
| Sentiment Analysis | 0.6210 | 0.7522 |

## Why MiniLM?

MiniLM performed better at both topic classification and sentiment prediction. Its sentence embeddings capture semantic relationships between differently worded customer reviews, making it more suitable for varied feedback.

TF-IDF is simpler, faster and requires fewer computational resources, but its lower evaluation scores made MiniLM the preferred option.

The application retains TF-IDF as a baseline under Advanced Model Comparison for technical evaluation.

## Additional Improvements

After initial development and testing, the application was extended with:

- **Mixed-feedback analysis:** Separates positive and negative aspects where possible.
- **Multilingual handling:** Detects some Hindi/Hinglish feedback and offers Groq-powered translation before classification.
- **Evidence-backed weekly summaries:** Uses Groq with source validation and fallback summaries for unsupported claims.
- **Emerging issue detection:** Groups recurring negative complaints requiring further investigation.
- **Optional image feedback:** Supports user-confirmed AI image descriptions.

## Reliability and Deployment Considerations

The main dashboard uses local ML models and remains functional when Groq is unavailable. Temporary API failures are handled with limited retries, and weekly summaries have an evidence-based fallback.

MiniLM requires more memory and loading time than TF-IDF. Streamlit Community Cloud is suitable for demonstration, but its inactivity sleep behavior can delay startup. Always-on hosting and persistent caching would be preferable for production use.

The training data is synthetic, multilingual and image support have limited validation, and the classifier still makes mistakes on mixed or ambiguous feedback. Sensitive customer information should not be sent to external AI services without appropriate privacy safeguards.

## Final Decision

**Ship MiniLM as the default classifier while retaining TF-IDF for comparison.**

The current MVP demonstrates the required customer-feedback analysis workflow. Before production use, I would expand real-world evaluation, improve negative-complaint recall, monitor API costs and failures, and introduce stronger data-privacy and deployment controls.