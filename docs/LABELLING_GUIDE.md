# Human labelling guide — read before evaluation

**File:** `data/evaluation_to_verify.csv` · 120 synthetic, independently worded examples · labels initially suggested by AI.

1. Open the CSV in Excel or a spreadsheet editor. Read **every candidate row** you intend to verify. Do not use model predictions as the ground truth.
2. Assign **one primary topic**, based on the issue the customer is writing about. Topics: Delivery (shipping/tracking), Product Quality (working condition/materials), Pricing (price/promotions), Refunds (money returned), Customer Support (agents/help), Packaging (box/protective materials), Website/App (site/app/checkout interface), Payments (transaction/charges), Returns (sending item back/exchanges), Other (loyalty/membership/other themes).
3. Assign **one overall sentiment**: positive (satisfied/praising), negative (dissatisfied/complaining), neutral (fact-based or asking a question without clear judgment). In a mixed review, choose the **negative complaint** as the primary label and note the positive aspect separately in the UI explanation. Preserve the real text.
4. Correct the suggested `topic` and `sentiment` where necessary. Set `human_verified` to `yes` **only after you personally read and checked that row**. Verify at least **100 unique rows**, preferably all 120. Keep the original `id` and `text` unchanged; save as UTF-8 CSV.
5. Run `python -m src.evaluate` from the project root after installing `requirements-embeddings.txt`. Check the results table/confusion matrices. Open `outputs/mistakes_to_analyse.csv` and explain at least five *actual* remaining errors in the `why_the_model_failed` column.

All evaluation texts are distinct from the bundled training examples. The evaluation script also removes normalized exact duplicates from training defensively. Do not mark rows `yes` without manual verification, and do not report provisional scores as verified scores.
