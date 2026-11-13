# Customer Feedback Insights Analyzer

An AI-powered web application that helps businesses analyse customer reviews and support messages, identify common complaints, and understand customer sentiment.

The application uses Machine Learning and Large Language Models (LLMs) to turn customer feedback into useful insights through an interactive dashboard.

**Live Demo:** [Try the Application]([https://customerfeedbackanalyzer-ai.streamlit.app/])

## Features

- **CSV Upload:** Upload customer reviews or support messages for analysis.
- **Topic Classification:** Categorizes feedback into 10 business-related topics.
- **Sentiment Analysis:** Identifies positive, negative, and neutral feedback.
- **Urgent Issue Detection:** Flags potentially serious customer complaints.
- **Mixed Feedback Analysis:** Handles reviews containing both positive and negative experiences.
- **Emerging Issues:** Groups feedback that may represent new problems.
- **AI Weekly Summaries:** Generates summaries using an LLM, supported by original customer quotes.
- **Model Evaluation:** Compares traditional ML and embedding-based models using accuracy, F1 scores, and confusion matrices.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python, Pandas | Data processing |
| Streamlit, Plotly | Dashboard and visualizations |
| Scikit-learn | Machine learning |
| TF-IDF, MiniLM | Text classification approaches |
| Logistic Regression | Topic and sentiment prediction |
| K-Means | Issue clustering |
| Groq API | LLM-powered summaries |

## How It Works

1. Upload a CSV file containing customer feedback.
2. The application cleans and processes the data.
3. ML models classify feedback by topic and sentiment.
4. The dashboard displays trends, urgent issues, and possible new problems.
5. The LLM generates weekly summaries backed by actual customer quotes.

## Running Locally (Optional)

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

LLM summaries require a Groq API key configured securely through Streamlit secrets.

## Dataset and Evaluation

The project uses synthetic customer feedback for demonstration. Two classification approaches—TF-IDF and MiniLM embeddings—are compared using a separate evaluation dataset.

Performance is measured using accuracy, macro F1 scores, and confusion matrices.

## Limitations

The application is designed as an MVP. Classification accuracy may vary on real customer feedback, and emerging issues or urgent complaints may require manual review.
