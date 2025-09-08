import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai
import warnings
import os
from difflib import get_close_matches
import sqlite3
import re
import tempfile
from pdfpreprocessing import pdf_to_csv_buffer,preprocess_bank_dataframe
warnings.filterwarnings('ignore')
DB_FILE = "expense_data.db"
GEMINI_API_KEY = "API_KEY"  

class DataIngestionPipeline:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None

    def load_file(self, uploaded_file):
        try:
            self.raw_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(self.raw_data)} records")
            return True
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return False

    def _find_column(self, df, *name_groups):
        cols = [c.strip().lower() for c in df.columns]
        for names in name_groups:
            for n in names:
                if n in cols:
                    return n
            anchor = names[0]
            match = get_close_matches(anchor, cols, n=1, cutoff=0.75)
            if match:
                return match[0]
            for c in cols:
                if any(n in c for n in names):
                    return c
        return None

    def standardize_data(self):
        if self.raw_data is None or self.raw_data.empty:
            st.error("❌ No data detected in the file.")
            return False
        try:
            df = self.raw_data.copy()
            df.columns = [str(c).strip().lower() for c in df.columns]
            date_col = self._find_column(df, ['date', 'transaction date', 'posted date', 'time'])
            amt_col = self._find_column(df, ['amount', 'amt', 'transaction amount', 'value'])
            withdraw_col = self._find_column(df, ['withdrawal', 'debit', 'debits', 'withdrawals'])
            deposit_col = self._find_column(df, ['deposit', 'credit', 'credits', 'deposits'])
            desc_col = self._find_column(df, ['description', 'narration', 'details', 'merchant', 'note', 'particulars'])
            type_col = self._find_column(df, ['transaction type', 'type', 'txn type', 'direction'])

            if not date_col:
                st.error("❌ Could not detect a Date column.")
                st.write("Available columns:", df.columns.tolist())
                return False
            df['date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            for col in ['deposits', 'withdrawals']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('[,₹]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            if amt_col:
                amt = df[amt_col].astype(str).str.replace('[,₹]', '', regex=True)
                amt = amt.str.replace(r'\s*(CR|CREDIT)\s*', '', regex=True, case=False)
                amt = amt.str.replace(r'\s*(DR|DEBIT)\s*', '', regex=True, case=False)
                df['amount'] = pd.to_numeric(amt, errors='coerce')
            elif withdraw_col and deposit_col:
                df['withdrawal'] = pd.to_numeric(df[withdraw_col], errors='coerce').fillna(0)
                df['deposit'] = pd.to_numeric(df[deposit_col], errors='coerce').fillna(0)
                df['amount'] = df['deposit'] - df['withdrawal']
                df.loc[(df['deposit'] > 0) & (df['amount'] < 0), 'amount'] = df['deposit']
                df.loc[(df['withdrawal'] > 0) & (df['amount'] > 0), 'amount'] = -df['withdrawal']
            else:
                st.error("❌ Could not detect Amount/Withdraw/Deposit columns.")
                st.write("Available columns:", df.columns.tolist())
                return False

            df['transaction_type'] = df[type_col].astype(str).str.strip().str.title() if type_col else ''
            df['description'] = df[desc_col] if desc_col else ''

            income_kw = ['credit', 'salary', 'income', 'bonus', 'refund', 'dividend', 'interest', 'reversal', 'upi cr', 'upi credit','neft cr']
            expense_kw = ['debit', 'purchase', 'payment', 'pos', 'spent', 'charge', 'fee', 'emi', 'fuel', 'upi dr', 'upi debit','atm wdl',]

            df['is_income'] = df['amount'] > 0
            ttxt = df['transaction_type'].astype(str).str.lower() + ' ' + df['description'].astype(str).str.lower()
            
            df.loc[ttxt.str.contains('|'.join(income_kw), na=False), 'is_income'] = True
            df.loc[ttxt.str.contains('|'.join(expense_kw), na=False), 'is_income'] = False

            df.loc[df['is_income'], 'amount'] = df.loc[df['is_income'], 'amount'].abs()
            df.loc[~df['is_income'], 'amount'] = -df.loc[~df['is_income'], 'amount'].abs()

            df['month'] = df['date'].dt.to_period('M').astype(str)

            self.processed_data = df[['date', 'month', 'amount', 'transaction_type', 'description']].dropna(subset=['date', 'amount'])
            st.success(f"✅ Standardized {len(self.processed_data)} transactions")
            return True
        
        except Exception as e:
            st.error(f"❌ Error standardizing data: {str(e)}")
            return False

class TransactionCategorizer:
    def __init__(self, gemini_api_key, model_name="gemini-1.5-flash"):
        self.category_rules = {
            'Income': ['salary', 'bonus', 'income', 'refund', 'credit', 'dividend', 'interest', 'reversal'],
            'Food & Dining': ['restaurant', 'food', 'grocery', 'cafe', 'snack', 'swiggy', 'zomato'],
            'Travel & Transport': ['uber', 'bus', 'taxi', 'fuel', 'flight', 'hotel', 'metro', 'cab'],
            'Shopping': ['shopping', 'amazon', 'flipkart', 'mall', 'clothing', 'electronics','shop'],
            'Entertainment': ['movie', 'cinema', 'netflix', 'spotify', 'games', 'concert'],
            'Healthcare': ['hospital', 'pharmacy', 'doctor', 'lab', 'insurance'],
            'Utilities': ['electricity', 'internet', 'water', 'mobile', 'gas', 'broadband'],
            'Housing': ['rent', 'maintenance', 'repairs'],
            'Education': ['course', 'books', 'tuition'],
            'Investments': ['mutual fund', 'stock', 'brokerage', 'sip', 'buy'],
            'Transfers': ['atm', 'withdrawal', 'wallet', 'transfer', 'top-up'],
            'Others': []
        }
        self.cache = {}
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)

    def categorize_transaction_rule_based(self, description, ttype):
        text = f"{description}".lower()
        for cat, keywords in self.category_rules.items():
            if any(kw in text for kw in keywords):
                return cat, 0.85
        return 'Others', 0.35

    def categorize_with_gemini(self, description, ttype):
        key = (description, ttype)
        if key in self.cache:
            return self.cache[key]
        try:
            prompt = (
                "Classify personal finance transactions into ONE category:\n"
                f"{list(self.category_rules.keys())}.\n"
                f"Description: {description}\nType: {ttype}"
            )
            response = self.model.generate_content(prompt)
            cat = response.text.strip()
            if cat not in self.category_rules:
                cat = 'Others'
            self.cache[key] = (cat, 0.9)
            return cat, 0.9
        except Exception as e:
            if "quota" in str(e).lower():
                st.warning("⚠️ Gemini API quota exceeded. Using rule-based category fallback.")
                return 'Others', 0.4
            else:
                return 'Others', 0.5
    #it works like if the confidene score<0.5, then calls the Gemini model to categorize

    # def categorize_all_transactions(self, df):
    #     df = df.copy()
    #     if 'category' not in df.columns:
    #         df['category'] = ''
    #     existing = df['category'].astype(str).str.strip().str.lower()
    #     needs_cat = (existing == '') | (existing.isna()) | (existing.isin(['others', 'other', 'uncategorized', 'unknown']))
    #     categories, confidences = [], []
    #     for i, row in df.iterrows():
    #         if not needs_cat.loc[i]:
    #             categories.append(row['category'])
    #             confidences.append(1.0)
    #             continue
    #         c, cf = self.categorize_transaction_rule_based(row['description'], row['transaction_type'])
    #         if cf < 0.5 or c == 'Others':
    #             c, cf = self.categorize_with_gemini(row['description'], row['transaction_type'])
    #         categories.append(c)
    #         confidences.append(cf)
    #     df['category'] = categories
    #     df['confidence'] = confidences
    #     return df

    #It works like it calls both rule-based and gemini-based then categorize based on the highest cf
    def categorize_all_transactions(self, df):
        df = df.copy()
        if 'category' not in df.columns:
            df['category'] = ''
        existing = df['category'].astype(str).str.strip().str.lower()
        needs_cat = (existing == '') | (existing.isna()) | (existing.isin(['others', 'other', 'uncategorized', 'unknown']))
        categories, confidences = [], []
        for i, row in df.iterrows():
            if not needs_cat.loc[i]:
                categories.append(row['category'])
                confidences.append(1.0)
                continue

            rule_cat, rule_cf = self.categorize_transaction_rule_based(row['description'], row['transaction_type'])
            gemini_cat, gemini_cf = self.categorize_with_gemini(row['description'], row['transaction_type'])
            if rule_cf >= gemini_cf:
                categories.append(rule_cat)
                confidences.append(rule_cf)
            else:
                categories.append(gemini_cat)
                confidences.append(gemini_cf)
        df['category'] = categories
        df['confidence'] = confidences
        return df


class AnalyticsEngine:
    def __init__(self, df):
        self.df = df.copy()
        if 'month' not in self.df.columns:
            self.df['month'] = self.df['date'].dt.to_period('M').astype(str)

    def monthly_rollups(self):
        return self.df.groupby('month').agg(
            income=('amount', lambda x: x[x > 0].sum()),
            expense=('amount', lambda x: -x[x < 0].sum()),
            net=('amount', 'sum')
        ).reset_index()

    def monthly_summary(self):
        out = self.df[self.df['category'] != 'Income'].copy()
        summary = out.groupby(['month', 'category'], as_index=False).agg(
            total_amount=('amount', 'sum'),
            transactions=('amount', 'count')
        )
        return summary

class DashboardGenerator:
    def __init__(self, df):
        self.df = df

    def spending_overview_by_category(self):
        cat_sp = self.df[(self.df['amount'] < 0) & (self.df['category'] != 'Income')]
        summary = cat_sp.groupby('category')['amount'].sum().abs().reset_index()
        chart = px.pie(summary, values='amount', names='category', title="Spending by Category")
        return chart, summary

    def spending_overview_by_transaction_type(self):
        tx_sp = self.df[self.df['amount'] < 0].groupby('transaction_type')['amount'].sum().abs().reset_index()
        chart = px.pie(tx_sp, values='amount', names='transaction_type', title="Spending by Transaction Type")
        return chart, tx_sp

    def monthly_income_vs_expense(self):
        tmp = self.df.copy()
        if 'month' not in tmp.columns:
            tmp['month'] = tmp['date'].dt.to_period('M').astype(str)
        roll = tmp.groupby('month').agg(
            Income=pd.NamedAgg(column='amount', aggfunc=lambda x: x[x > 0].sum()),
            Expense=pd.NamedAgg(column='amount', aggfunc=lambda x: -x[x < 0].sum())
        ).reset_index()
        chart = px.bar(roll, x='month', y=['Income', 'Expense'], barmode='group', title="Monthly Income vs Expense")
        return chart, roll

    def monthly_summary_by_category_chart(self, analytics_engine):
        summary = analytics_engine.monthly_summary()
        summary['abs_amount'] = summary['total_amount'].abs()
        chart = px.bar(
            summary,
            x='month',
            y='abs_amount',
            color='category',
            barmode='stack',
            title="Monthly Summary by Category ",
            labels={'abs_amount': 'Amount (₹)', 'month': 'Month', 'category': 'Category'}
        )
        return chart, summary

class RAGQueryHandler:
    def __init__(
        self, 
        df, 
        category_summary, 
        transaction_type_summary, 
        income_expense_summary, 
        monthly_category_summary,
        gemini_api_key,
        model_name='gemini-1.5-flash'):
        self.df = df.copy()
        self.category_summary = category_summary
        self.transaction_type_summary = transaction_type_summary
        self.income_expense_summary = income_expense_summary
        self.monthly_category_summary = monthly_category_summary
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def answer_question(self, question):
        q_lower = question.lower()

        if "spending by category" in q_lower or "which category" in q_lower or "top category" in q_lower:
            top = self.category_summary.sort_values('amount', ascending=False).iloc[0]
            return f"Your highest spending category was {top['category']} (₹{top['amount']:,.2f}) this period."
        
        if "transaction type" in q_lower or "mode of payment" in q_lower:
            top = self.transaction_type_summary.sort_values('amount', ascending=False).iloc[0]
            return f"Highest spend by transaction type: {top['transaction_type']} (₹{top['amount']:,.2f})."
        
        if "income" in q_lower or "expense" in q_lower or "savings" in q_lower:
            latest = self.income_expense_summary.sort_values('month', ascending=False).iloc[0]
            net_savings = latest['Income'] - latest['Expense']
            return (f"For {latest['month']}, your income was ₹{latest['Income']:,.2f}, expenses were "
                    f"₹{latest['Expense']:,.2f}, net savings ₹{net_savings:,.2f}.")
        
        mt_match = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december) \d{4}", q_lower)
        if mt_match:
            period = mt_match.group().title()
            out = self.monthly_category_summary[self.monthly_category_summary['month'] == period]
            if not out.empty:
                rows = [f"{row['category']}: ₹{row['total_amount']:,.2f}" for _, row in out.iterrows()]
                return f"Spending breakdown for {period}:\n" + "\n".join(rows)
        
        prompt = (
            f"Charts Summary:\n"
            f"Category Spend: {self.category_summary.to_dict('records')}\n"
            f"Transaction Type Spend: {self.transaction_type_summary.to_dict('records')}\n"
            f"Income vs Expense Monthly: {self.income_expense_summary.to_dict('records')}\n"
            f"Monthly Category Summary: {self.monthly_category_summary.to_dict('records')}\n"
            f"User question: {question}\n"
            "Please answer using the chart information provided and dont use $ symbol."
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"


def save_df_to_db(df, db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    conn.close()

def load_df_from_db(db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql('SELECT * FROM transactions', conn)
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def main():
    st.title("💰 AI Financial Expense Analyzer")
    # df = load_df_from_db()
    data_loaded_from_db = False

    uploaded = st.file_uploader("Upload your CSV or PDF", type=['csv', 'pdf'])
    if uploaded:
        pipeline = DataIngestionPipeline()
        if uploaded.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(uploaded.read())
                tmp_pdf.flush()
                extracted_df = pdf_to_csv_buffer(tmp_pdf.name)
            os.unlink(tmp_pdf.name)

            if extracted_df is not None and not extracted_df.empty:
                if 'Particulars' in extracted_df.columns:
                    extracted_df = extracted_df.rename(columns={'Particulars': 'description'})
                cleaned_df = preprocess_bank_dataframe(extracted_df)
                pipeline.raw_data = cleaned_df
                if pipeline.standardize_data():
                    df = pipeline.processed_data.copy()
                    df['transaction_type'] = df['transaction_type'].replace({'Withdrawal': 'Debit'})
                    if 'description' in df.columns:
                        mask = df['description'].astype(str).str.contains('UPI', case=False, na=False)
                        df.loc[mask, 'transaction_type'] = 'UPI'
                    categorizer = TransactionCategorizer(gemini_api_key=GEMINI_API_KEY)
                    df = categorizer.categorize_all_transactions(df)
                    save_df_to_db(df)
                    data_loaded_from_db = True
                else:
                    st.error("❌ Failed to standardize data converted from PDF.")
            else:
                st.error("❌ Failed to convert PDF to CSV or PDF contained no transactions.")

        elif uploaded.type == "text/csv":
            if pipeline.load_file(uploaded) and pipeline.standardize_data():
                df = pipeline.processed_data.copy()
                df['transaction_type'] = df['transaction_type'].replace({'Withdrawal': 'Debit'})
                if 'description' in df.columns:
                    mask = df['description'].astype(str).str.contains('UPI', case=False, na=False)
                    df.loc[mask, 'transaction_type'] = 'UPI'
                categorizer = TransactionCategorizer(gemini_api_key=GEMINI_API_KEY)
                df = categorizer.categorize_all_transactions(df)
                save_df_to_db(df)
                data_loaded_from_db = True
            else:
                st.error("❌ Failed to load or standardize CSV.")
        else:
            st.error("❌ Unsupported file type.")

    if data_loaded_from_db:
        analytics = AnalyticsEngine(df)
        dash = DashboardGenerator(df)
        cat_chart, category_summary = dash.spending_overview_by_category()
        tx_chart, transaction_type_summary = dash.spending_overview_by_transaction_type()
        income_chart, income_expense_summary = dash.monthly_income_vs_expense()
        monthly_cat_chart, monthly_category_summary = dash.monthly_summary_by_category_chart(analytics)

        st.sidebar.header("💬 Ask your expense data questions")
        user_question = st.sidebar.text_input("Enter your question:", key="rag_input")
        if user_question:
            rag_handler = RAGQueryHandler(
                df,
                category_summary,
                transaction_type_summary,
                income_expense_summary,
                monthly_category_summary,
                gemini_api_key=GEMINI_API_KEY)
            with st.spinner("⌛ Generating answer..."):
                answer = rag_handler.answer_question(user_question)
            st.sidebar.markdown(f"**Answer:** {answer}")

        total_tx = len(df)
        total_income = df.loc[df['amount'] > 0, 'amount'].sum()
        total_expense = -df.loc[df['amount'] < 0, 'amount'].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", f"{total_tx:,}")
        c2.metric("Total Income", f"₹{total_income:,.2f}")
        c3.metric("Total Expenses", f"₹{total_expense:,.2f}")

        st.plotly_chart(cat_chart, use_container_width=True)
        st.plotly_chart(tx_chart, use_container_width=True)
        st.plotly_chart(income_chart, use_container_width=True)
        st.plotly_chart(monthly_cat_chart, use_container_width=True)
        roll = analytics.monthly_rollups().sort_values('month')
        cats = df[df['amount'] < 0].copy()
        cats['abs_spend'] = -cats['amount']
        if 'month' not in cats.columns:
            cats['month'] = cats['date'].dt.to_period('M').astype(str)
        cat_m = cats.groupby(['month', 'category'], as_index=False)['abs_spend'].sum()

        st.subheader("Insights & Alerts")
        for month in roll['month'].unique():
            with st.expander(f"🔍 {month} Alerts & Insights", expanded=False):
                cur = roll[roll['month'] == month].iloc[0]
                prev_idx = roll[roll['month'] == month].index[0] - 1
                if prev_idx >= 0:
                    prev = roll.iloc[prev_idx]
                    if prev['expense'] > 0:
                        pc = (cur['expense'] - prev['expense']) / prev['expense'] * 100
                        diff = cur['expense'] - prev['expense']
                        direction = "more" if pc > 0 else "less"
                        st.info(
                            f"📈 You spent {abs(pc):.1f}% {direction} than {prev['month']} " +
                            f"(actual difference: ₹{abs(diff):,.0f})."
                        )

                        
                cm = cat_m[cat_m['month'] == month].sort_values('abs_spend', ascending=False)
                if not cm.empty and cm.iloc[0]['abs_spend'] > 0:
                    st.info(f"💡 Highest spend category: {cm.iloc[0]['category']} (₹{cm.iloc[0]['abs_spend']:,.0f})")
                total = cur['income'] + cur['expense']
                if cur['income'] > 0 and total > 0:
                    savings_rate = (cur['income'] - cur['expense']) / cur['income'] * 100
                    st.info(f"💾 Savings rate: {savings_rate:.1f}%")
                subscription_keywords = ['subscription', 'netflix', 'spotify', 'amazon prime', 'prime', 'hulu', 'disney+', 'apple music', 'youtube premium', 'saas', 'membership']

                subs = df[df['description'].str.contains('|'.join(subscription_keywords), case=False, na=False)].copy()
                if 'month' not in subs.columns:
                    subs['month'] = subs['date'].dt.to_period('M').astype(str)
                subs_m = subs.groupby('month')['amount'].sum().abs()

                if month in subs_m.index:
                    current_subs = subs_m[month]
                    prev_subs = subs_m.get(roll.iloc[prev_idx]['month'], 0)
                    if prev_subs > 0:
                        pct_change_subs = (current_subs - prev_subs) / prev_subs * 100
                        if pct_change_subs >= 0 and pct_change_subs<100:
                            
                            st.warning(f"🚨 Your subscriptions went up {pct_change_subs:.0f}% this month!")

    else:
        st.info("Please upload a CSV or PDF file containing your transactions to get started.")

if __name__ == "__main__":
    main()
