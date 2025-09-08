💰 AI Financial Expense Analyzer

An AI-powered financial expense analyzer built with Streamlit, Plotly, and Google Gemini API.
It helps you upload bank statements (CSV or PDF), preprocess them, categorize transactions, and visualize insights with interactive dashboards.

🚀 Features

📂 Upload CSV or PDF of your bank transactions

🔄 Automatic preprocessing & standardization of transaction data

🤖 AI-powered categorization (rule-based + Gemini model fallback)

📊 Interactive dashboards with Plotly

    Spending by category
    Spending by transaction type
    Monthly income vs. expense
    Monthly breakdown by category
    
💬 RAG-based Q&A about your expenses using Gemini

📈 Insights & alerts

    Savings rate
    Spending growth vs. previous month
    Subscription spending alerts

🛠️ Tech Stack

Python (pandas, sqlite3, re, tempfile, os)

Streamlit – Web app interface

Plotly Express – Charts & visualizations

pdfplumber – Extracts tables from PDFs

Google Gemini API – AI transaction categorization & Q&A

🧩 How It Works

File Upload & Preprocessing

        Upload a CSV or PDF bank statement.

        CSVs are read directly using pandas.

        PDFs are parsed via pdfplumber → cleaned into a DataFrame using pdfpreprocessing.py.

Data Standardization

        Detects key columns: date, amount, deposits, withdrawals, description, transaction type.

        Normalizes amounts (removes currency symbols, handles CR/DR).

        Handles income vs. expenses automatically.

Transaction Categorization

        Rule-based categorization (e.g., keywords: salary, fuel, shopping, grocery).

        If confidence score is low → fallback to Gemini AI model for categorization.

        Each transaction is tagged with a category + confidence score.

Analytics & Dashboard

        Monthly rollups: income, expenses, net savings.

        Spending breakdown: by category & by transaction type.

        Trend charts: monthly income vs. expense.

        Stacked bar charts: monthly spend by category.

RAG-based Question Answering

        Ask natural language questions like:

            “What was my top spending category in July 2024?”

            “How much did I save last month?”

            “Which transaction type had the highest spend?”

            Answers are generated using transaction summaries + Gemini AI.

Insights & Alerts

        📈 Compare spending with the previous month.

        💡 Identify highest spending category each month.

        💾 Show savings rate as a percentage.

        🚨 Subscription spending alerts.
