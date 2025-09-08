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
  CSVs are read directly.
  PDFs are parsed via pdfplumber → cleaned into a DataFrame.

Data Standardization
  Detects date, amount, deposits, withdrawals, description, and type columns.
  Normalizes amounts, handles credits/debits.

Transaction Categorization
  Rule-based categorization (keywords like salary, fuel, shopping).
  If confidence < 0.5 → fallback to Gemini model.

Analytics & Dashboard
  Rollups of income, expenses, and savings.
  Spending breakdown by category & type.

RAG Question Answering
  Ask natural-language questions like:
    “What was my top spending category in July 2024?”
    “How much did I save last month?”
    “Which transaction type had the highest spend?”
