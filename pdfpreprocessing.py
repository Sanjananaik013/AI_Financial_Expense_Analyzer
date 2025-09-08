import pdfplumber
import pandas as pd

def pdf_to_csv_buffer(file_path):
    all_rows = []
    header = None
    possible_headers = set()
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                page_header = [str(cell).strip() if cell else "" for cell in table[0]]
                possible_headers.add(tuple(page_header))
                if header is None:
                    header = page_header
                for row in table[1:]:
                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                    if cleaned_row != header and any(cleaned_row):
                        if tuple(cleaned_row) not in possible_headers:
                            all_rows.append(cleaned_row)
                        else:
                            continue
    if header is None or not all_rows:
        return pd.DataFrame()
    all_rows = [row for row in all_rows if row != header]
    df = pd.DataFrame(all_rows, columns=header)
    df = df.dropna(how="all")
    return df


def preprocess_bank_dataframe(df):
    df.columns = df.columns.str.lower().str.strip()
    df = df[df['date'].notnull() & (df['date'].astype(str).str.strip() != "")]
    df = df[df['description'].str.lower() != 'opening balance']
    for col in ['deposits', 'withdrawals']:
        if col in df.columns:
            df[col] = df[col].replace('', '0').astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0
    
    df['amount'] = df['deposits'] - df['withdrawals']
    df.loc[(df['deposits'] > 0) & (df['amount'] < 0), 'amount'] = df['deposits']
    df.loc[(df['withdrawals'] > 0) & (df['amount'] > 0), 'amount'] = -df['withdrawals']
    
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['transaction_type'] = ''
    df['description'] = df['description']
    df = df[['date', 'month', 'amount', 'transaction_type', 'description']]
    df = df.dropna(subset=['date', 'amount'])
    return df