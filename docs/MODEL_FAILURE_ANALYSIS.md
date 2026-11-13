# Five model failures — provisional diagnostic examples

**Status: NOT FINAL / NOT HUMAN-VERIFIED.** These are *observed* errors from running the **classic TF-IDF baseline** against the 120 AI-drafted seed labels. The labels must be independently reviewed by the candidate, and the actual final errors must be checked again after running **both** methods. The reasons below are hypotheses based on the input wording and training design; they are not formal feature-attribution results.

| Review ID | Task | Draft actual label | Baseline prediction | Likely explanation and improvement |
| --- | --- | --- | --- | --- |
| **E0005** | Topic | Delivery | Product Quality | The express-shipping sentence uses unfamiliar phrasing and generic dissatisfaction. Add more shipping-speed paraphrases to training; validate performance on held-out delivery cases. |
| **E0011** | Topic | Delivery | Website/App | The shipping question has neutral language and lacks words that dominate the baseline's templated delivery examples. Improve diversity of question-style training records. |
| **E0013** | Topic | Product Quality | Delivery | The malfunctioning charger is expressed as "stopped working" rather than the template's usual quality keywords. Add device-failure and accessory terms from independent training examples. |
| **E0025** | Topic | Pricing | Product Quality | The advertised-price complaint is semantically clear but phrased differently from the original price templates. Pretrained embeddings may generalize better across synonyms; check verified scores. |
| **E0032** | Topic | Pricing | Product Quality | The text praises both fair pricing and product quality. A single-label classifier can emphasize the wrong aspect. Make primary-topic annotation rules explicit; explore aspect-aware classification later. |
| **E0006** | Sentiment | Positive | Negative | The arrival-early compliment contains delivery vocabulary that co-occurs with complaints in templated training data. Balance happy delivery examples and examine per-class sentiment confusion. |

**How to finalize:** Hand-verify at least 100 labels, run `python -m src.evaluate`, open `outputs/mistakes_to_analyse.csv`, find at least five genuine errors from the final predictions, and replace/revise this draft with the final observed mistakes and their plausible causes. **Do not submit the draft as evidence of completed human evaluation.**
