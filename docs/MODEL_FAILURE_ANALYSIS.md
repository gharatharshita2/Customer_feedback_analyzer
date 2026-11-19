# Model Failure Analysis – Customer Feedback Insights Analyzer

## Evaluation Overview

The TF-IDF and MiniLM classifiers were evaluated using the same 100 human-reviewed customer feedback samples.

MiniLM achieved **85% topic classification accuracy** and **74% sentiment accuracy**. Despite performing better overall than TF-IDF, it made several errors involving similar business categories, mixed feedback and neutral questions.

The following five mistakes were taken from the saved MiniLM prediction results.

## Five Actual Model Failures

### Mistake 1: Packaging vs Product Quality

**Review ID:** E0063  
**Correct Label:** Packaging  
**Predicted Label:** Product Quality

**Problem:** The customer reported that the product worked, but its outer packaging was crushed and soaked.

**Likely Reason:** The model associated product-related words with product quality rather than identifying damaged packaging as the main complaint.

**Suggested Improvement:** Include more examples distinguishing packaging damage from defects in the actual product.

### Mistake 2: Payments vs Returns

**Review ID:** E0088  
**Correct Label:** Payments  
**Predicted Label:** Returns

**Problem:** The customer reported that a UPI payment was successful, but the store considered the order unpaid.

**Likely Reason:** The model failed to correctly interpret the payment-status discrepancy.

**Suggested Improvement:** Add more examples involving UPI, failed transactions, payment confirmation and order-payment mismatches.

### Mistake 3: Returns vs Refunds

**Review ID:** E0097  
**Correct Label:** Returns  
**Predicted Label:** Refunds

**Problem:** The customer reported repeated cancellation of a return pickup.

**Likely Reason:** Returns and refunds often appear in related customer-service situations, making them difficult to distinguish.

**Suggested Improvement:** Add clearer training examples differentiating return pickups, exchanges and refund processing.

### Mistake 4: Mixed Sentiment Misclassification

**Review ID:** E0001  
**Correct Sentiment:** Negative  
**Predicted Sentiment:** Positive

**Problem:** The customer praised the shoes but complained that delivery arrived five days late.

**Likely Reason:** Positive product-related wording may have outweighed the delivery complaint in the sentiment prediction.

**Suggested Improvement:** Include more mixed-sentiment examples and improve aspect-level handling.

### Mistake 5: Neutral Question Classified as Negative

**Review ID:** E0022  
**Correct Sentiment:** Neutral  
**Predicted Sentiment:** Negative

**Problem:** The customer asked whether a bottle contained BPA.

**Likely Reason:** The model may have associated the safety-related term with negative customer feedback, even though the message was only a question.

**Suggested Improvement:** Train on more neutral product-information and safety-related inquiries.

## Additional Challenges Found During Application Testing

### 1. Hindi and Hinglish Feedback

**Observed problem:** The English-trained classifier incorrectly labelled two negative Hindi/Hinglish delivery complaints as positive when analysing their original text.

**Improvement implemented:** Added basic language detection and optional Groq-powered translation before classification.

**Retest result:** Both complaints were classified as negative after translation.

**Remaining limitation:** The feature was tested on only a few examples. Translation accuracy and multilingual performance require broader evaluation.

### 2. Groq API Availability and Rate Limits

**Challenge:** External LLM calls can fail because of request limits, network timeouts or temporary server errors.

**Improvement implemented:** Added a limited retry mechanism and an evidence-only fallback for weekly summaries. Translation failures also allow the original-text analysis to continue.

**Remaining limitation:** Extended outages, API quotas and repeated requests across browser sessions still need stronger production-level handling.

### 3. Unsupported AI Summary Claims

**Observed problem:** In one weekly-summary test, only five of six generated claims passed the application's evidence checks.

**Improvement implemented:** Unsupported claims were rejected and replaced with source-backed evidence.

**Remaining limitation:** Passing source-ID and quote checks does not guarantee that every interpretation is semantically correct.

### 4. Known Issues vs Emerging Issues

**Observed challenge:** Clustering could group many similar loyalty-point complaints, but repetition alone could not establish whether the issue was genuinely new.

**Improvement implemented:** Added conservative candidate detection based on repeated negative feedback, with clear review status and supporting source examples. Business-rule inputs were experimented with but later removed from the client-facing dashboard.

**Remaining limitation:** Emerging issue clusters still require business review.

### 5. Image Feedback

**Challenge:** An image can show visible damage without proving what happened or how the customer feels.

**Improvement implemented:** Added optional AI image descriptions that users must confirm or edit before analysis.

**Remaining limitation:** Image descriptions are not treated as original customer quotes, and image-analysis accuracy has not been comprehensively evaluated.

## Conclusion

The project was improved by testing real prediction mistakes and examining how the system behaves outside its standard English CSV workflow.

Future work should focus on more diverse training data, improved mixed-sentiment classification, broader multilingual testing, persistent API-response caching and evaluation using real customer feedback.

The current application remains an MVP, with original source evidence and human review important for business decisions.