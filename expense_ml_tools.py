import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans


# ==============================
# 1. LOAD & CLEAN THE DATA
# ==============================

def load_expense_data(file_path: str = "expenses.xlsx") -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Expected columns (case-insensitive in Excel):
    # date, type, comments, rate, mode_of_transaction, monthly_total (optional)

    # Drop monthly total rows if present
    if "monthly_total" in df.columns:
        df = df[df["monthly_total"].isna()]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Rate as numeric
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0)

    # Ensure 'type' and 'comments' columns exist
    if "type" not in df.columns:
        df["type"] = ""

    if "comments" not in df.columns:
        df["comments"] = ""

    # Fill NaNs in strings
    df["type"] = df["type"].fillna("").astype(str)
    df["comments"] = df["comments"].fillna("").astype(str)

    # Month helpers
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year

    # Mode of transaction
    if "mode_of_transaction" in df.columns:
        df["mode_of_transaction"] = df["mode_of_transaction"].fillna("Unknown").astype(str)
    else:
        df["mode_of_transaction"] = "Unknown"

    return df


# ==========================================
# 2. SMART CATEGORY SUGGESTION (NLP MODEL)
# ==========================================

def train_category_suggestion_model(df: pd.DataFrame):
    """
    Train a text classifier: Comments -> Type
    Returns a sklearn Pipeline.
    """
    train_df = df.copy()

    # Keep rows with non-empty type & comments
    train_df = train_df[
        (train_df["type"].str.strip() != "") &
        (train_df["comments"].str.strip() != "")
    ]

    if train_df.empty:
        print("Not enough labeled data (Type & Comments) to train category model.")
        return None

    X = train_df["comments"]
    y = train_df["type"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    model.fit(X, y)
    print(f"Category suggestion model trained on {len(train_df)} records.")
    return model


def demo_category_suggestions(df: pd.DataFrame, model):
    if model is None:
        return

    print("\n=== SMART CATEGORY SUGGESTIONS (DEMO) ===")
    example_comments = [
        "Tea + Samosa",
        "Bus ticket to office",
        "Donation to mosque",
        "Recharge for mobile",
        "New shirt from mall",
    ]

    for c in example_comments:
        pred = model.predict([c])[0]
        print(f"Comment: {c:40s} -> Suggested Type: {pred}")

    # Also suggest for any rows with missing/blank type (if any)
    missing_type = df[df["type"].str.strip() == ""]
    if not missing_type.empty:
        print("\nRows with empty 'Type' - suggested categories:")
        for _, row in missing_type.iterrows():
            comment = row["comments"]
            pred = model.predict([comment])[0]
            print(f"{row['date'].date()} | '{comment}' -> {pred}")


# =====================================
# 3. MONTHLY EXPENSE FORECASTING
# =====================================

def forecast_monthly_spending(df: pd.DataFrame, months_ahead: int = 1):
    """
    Use simple Linear Regression on month index to predict future spending.
    """
    monthly = df.groupby("month")["rate"].sum().sort_index()
    if len(monthly) < 2:
        print("Not enough months of data to forecast.")
        return

    # Convert month labels to indices 0..n-1
    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly.values

    reg = LinearRegression()
    reg.fit(X, y)

    # Predict next N months
    future_indices = np.arange(len(monthly), len(monthly) + months_ahead).reshape(-1, 1)
    preds = reg.predict(future_indices)

    print("\n=== MONTHLY SPENDING FORECAST ===")
    for i, p in enumerate(preds, start=1):
        print(f"Forecast for month +{i}: ~ {p:.2f}")

    # Optional: visualize historical + regression line
    plt.figure(figsize=(8, 4))
    plt.scatter(X.flatten(), y, label="Actual")
    all_indices = np.arange(len(monthly) + months_ahead).reshape(-1, 1)
    all_preds = reg.predict(all_indices)
    plt.plot(all_indices.flatten(), all_preds, label="Trend/Forecast")
    plt.title("Monthly Spending & Forecast Trend")
    plt.xlabel("Month Index (0 = first month in data)")
    plt.ylabel("Total Spending")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================
# 4. ANOMALY / UNUSUAL EXPENSE DETECTION
# =====================================

def detect_expense_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag unusual transactions using z-scores on the 'rate' column.
    """
    rates = df["rate"]
    if rates.std() == 0:
        print("No variation in 'rate' to compute anomalies.")
        return pd.DataFrame()

    df = df.copy()
    df["z_score"] = (df["rate"] - rates.mean()) / rates.std()

    anomalies = df[df["z_score"].abs() >= z_threshold]

    print(f"\n=== ANOMALOUS / UNUSUAL EXPENSES (|z| >= {z_threshold}) ===")
    if anomalies.empty:
        print("No strong anomalies detected.")
    else:
        for _, row in anomalies.iterrows():
            print(
                f"{row['date'].date()} | {row['type']:10s} | "
                f"{str(row['comments'])[:30]:30s} | Amount: {row['rate']:7.2f} | z={row['z_score']:.2f}"
            )

    return anomalies


# =====================================
# 5. DAILY SPENDING CLUSTERING (KMEANS)
# =====================================

def cluster_daily_spending(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Cluster days based on how money is distributed across categories (Type).
    """
    # Create a pivot: one row per day, columns = types, values = sum of rate
    daily = df.pivot_table(
        index="date",
        columns="type",
        values="rate",
        aggfunc="sum",
        fill_value=0
    )

    if daily.shape[0] < n_clusters:
        print(f"Not enough days ({daily.shape[0]}) for {n_clusters} clusters.")
        return daily

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    daily["cluster"] = kmeans.fit_predict(daily)

    print(f"\n=== DAILY SPENDING CLUSTERS (k={n_clusters}) ===")
    print(daily["cluster"].value_counts().sort_index())

    # Optional: plot total spending per cluster
    daily["total_spent"] = daily.drop(columns=["cluster"]).sum(axis=1)
    cluster_summary = daily.groupby("cluster")["total_spent"].mean()

    plt.figure(figsize=(6, 4))
    plt.bar(cluster_summary.index.astype(str), cluster_summary.values)
    plt.title("Average Daily Spending per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Average Daily Spend")
    plt.tight_layout()
    plt.show()

    return daily


# ==============================
# MAIN ENTRY POINT
# ==============================

if __name__ == "__main__":
    print("Loading data from expenses.xlsx ...")
    df = load_expense_data("Expense.xlsx")

    # 1) Smart category suggestion
    model = train_category_suggestion_model(df)
    demo_category_suggestions(df, model)

    # 2) Monthly forecasting (next 1 month)
    forecast_monthly_spending(df, months_ahead=1)

    # 3) Anomaly detection
    anomalies_df = detect_expense_anomalies(df, z_threshold=2.0)

    # 4) Daily spending clustering
    clustered_daily = cluster_daily_spending(df, n_clusters=3)

    print("\nDone.")
