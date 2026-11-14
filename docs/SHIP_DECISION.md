# Shipping Decision – Customer Feedback Insights Analyzer

## Recommended Approach

I recommend **MiniLM Embeddings with Logistic Regression** for customer feedback classification because it performed better than TF-IDF in the evaluation.

## Evaluation Results

The models were evaluated using 100 human-verified customer feedback samples.

| Task | TF-IDF Accuracy | MiniLM Accuracy |
|---|---|---|
| Topic Classification | 51% | 85% |
| Sentiment Analysis | 63% | 74% |

## Why MiniLM?

MiniLM captures the meaning and context of customer feedback more effectively than TF-IDF. It provides better accuracy when handling different expressions and complex customer reviews.

Although TF-IDF is simpler and faster, MiniLM demonstrated stronger overall classification performance.

## Production Considerations

Before deployment, I would conduct additional testing on real-world feedback, monitor classification errors, and optimize model performance and resource usage.

## Conclusion

Based on the evaluation results, **MiniLM with Logistic Regression is my preferred approach** because of its higher classification accuracy and better understanding of customer feedback.
