# Customer Feedback Insights Analyzer

An AI-powered web application that helps businesses analyse customer reviews and support messages, identify common complaints, and understand customer sentiment.

The application uses Machine Learning and Large Language Models (LLMs) to turn customer feedback into useful insights through an interactive dashboard.

**Live Demo:** [Try the Application](https://customerfeedbackanalyzer-ai.streamlit.app/)

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

## Dataset and Evaluation

The project uses synthetic customer feedback and was evaluated using **100 human-verified samples**.

### Evaluation Results

| Model | Topic Accuracy | Sentiment Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression | 51% | 63% |
| MiniLM + Logistic Regression | 85% | 74% |

MiniLM achieved better performance and is the recommended approach for further production validation.

### Documentation

- [Evaluation Results](outputs/evaluation_results.json)
- [Five Analysed Mistakes](docs/MODEL_FAILURE_ANALYSIS.md)
- [Shipping Decision](docs/SHIP_DECISION.md)

## Project Walkthrough

**Video:** [Recording Link](https://drive.google.com/file/d/1ZaT0az4tgM3KfANPs7Ssbj1FBWaQOIMk/view?usp=sharing)

