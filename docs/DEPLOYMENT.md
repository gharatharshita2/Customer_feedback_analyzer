# Deploy as a one-link demo (no reviewer installations)

This is the recommended submission route. The app is hosted on **Streamlit Community Cloud** and weekly LLM summaries use a **Groq-hosted model**, not the project owner's local Ollama instance.

## Owner setup (once)

1. Create a free GroqCloud account at https://console.groq.com/keys and generate an API key. Check the current Free plan quota and model availability before publishing; no payment is necessary for usage within the free allowance.
2. Create a GitHub repository and push this project's source code, data, docs, tests, `.streamlit/config.toml`, and `requirements.txt`. Ensure **no secrets** are committed. The `.gitignore` already excludes `.streamlit/secrets.toml`.
3. Visit https://share.streamlit.io, choose **Create app**, select the repository/branch, and set the main file to `app.py`.
4. In **Advanced settings > Secrets**, add:

   ```toml
   GROQ_API_KEY = "your_real_key_here"
   ```

   Paste the real key only into Streamlit's private secrets interface, **never** README, GitHub, screenshots, or chat.
5. Deploy. The platform installs Python requirements server-side; reviewers do not install Python, MiniLM, Ollama, or the Groq client.
6. Test the public site with the included `data/sample_feedback_upload.csv`. Verify both classification methods, a weekly hosted LLM summary (at least one generated claim accepted), original quote links, mixed reviews, and the CSV error case.
7. Record the required 3–5 minute demonstration. Send the website URL and GitHub URL. Include verified evaluation metrics and at least five analysed mistakes after manually verifying 100 evaluation labels.

## Important guardrails

- When the Groq key is missing, the dashboard still functions and shows **extractive** evidence-only summaries, clearly marked. This is **not** a successful LLM demonstration.
- Every generated summary claim is checked against source topic, quote wording and real evidence IDs. Unsafe/unverifiable claims are replaced with extractive counts and quotations.
- The app sends selected negative customer quotes to Groq's hosted API. For this assessment use **synthetic/public feedback only**; do not upload confidential customer information without an appropriate privacy review.
- Hosting and free API quotas may change; if the provider returns a rate limit/error, the app keeps the safe extractive view instead of inventing a summary.
- Python 3.12 is recommended in Streamlit's **Advanced settings** for ML dependencies. The first MiniLM run downloads model weights onto the Streamlit server (not to reviewers' computers), and may use more resources.
- Evaluation results are produced by `python -m src.evaluate` after human label verification. To display them on the hosted site, commit the generated `outputs/evaluation_results.json` to GitHub **after** validating its results. `.gitignore` excludes this generated JSON by default; use `git add -f outputs/evaluation_results.json` when ready. Never claim that draft labels are human verified.

Official docs: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy and https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
