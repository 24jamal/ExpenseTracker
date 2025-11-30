import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans


# ==============================
# DATA LOADING & CLEANING
# ==============================

@st.cache_data
def load_expense_data(file_path: str = "expenses.xlsx") -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # Normalize column names
    df.columns = (
        df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
    )

    # Drop monthly total rows if present
    if "monthly_total" in df.columns:
        df = df[df["monthly_total"].isna()]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Rate as numeric
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0)

    # Ensure text columns
    if "type" not in df.columns:
        df["type"] = ""
    if "comments" not in df.columns:
        df["comments"] = ""

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


# ==============================
# AGGREGATIONS FOR CHARTS
# ==============================

def compute_basic_aggregations(df: pd.DataFrame):
    monthly_total = df.groupby("month")["rate"].sum().sort_index()
    category_total = df.groupby("type")["rate"].sum().sort_values(ascending=False)

    food_df = df[df["type"].str.lower().str.contains("food")]
    food_daily = (food_df.groupby("date")["rate"].sum()
                  if not food_df.empty else pd.Series(dtype=float))

    mode_total = df.groupby("mode_of_transaction")["rate"].sum().sort_values(ascending=False)

    charity_df = df[df["type"].str.lower().str.contains("charity")]
    charity_by_month = (charity_df.groupby("month")["rate"].sum().sort_index()
                        if not charity_df.empty else pd.Series(dtype=float))

    inv_df = df[df["type"].str.lower().str.contains("investment")]
    inv_by_month = (inv_df.groupby("month")["rate"].sum().sort_index()
                    if not inv_df.empty else pd.Series(dtype=float))

    essentials = ["food", "travel", "home", "health", "clothing", "recharge", "loan"]
    df_ess = df.copy()
    df_ess["is_essential"] = df_ess["type"].str.lower().apply(
        lambda t: "Essential" if any(e in t for e in essentials) else "Non-Essential"
    )
    ess_split = df_ess.groupby("is_essential")["rate"].sum()

    return {
        "monthly_total": monthly_total,
        "category_total": category_total,
        "food_daily": food_daily,
        "mode_total": mode_total,
        "charity_by_month": charity_by_month,
        "inv_by_month": inv_by_month,
        "ess_split": ess_split,
    }


# ==============================
# ML: CATEGORY SUGGESTION MODEL
# ==============================

def train_category_suggestion_model(df: pd.DataFrame):
    train_df = df[
        (df["type"].str.strip() != "") &
        (df["comments"].str.strip() != "")
    ]

    if train_df.empty or train_df["type"].nunique() < 2:
        return None

    X = train_df["comments"]
    y = train_df["type"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=300)),
    ])

    model.fit(X, y)
    return model


def get_category_suggestions_demo(model):
    if model is None:
        return None

    example_comments = [
        "Tea + Samosa",
        "Bus ticket to office",
        "Donation to mosque",
        "Recharge for mobile",
        "New shirt from mall",
    ]

    results = []
    for c in example_comments:
        pred = model.predict([c])[0]
        results.append((c, pred))
    return results


# ==============================
# ML: FORECASTING
# ==============================

def forecast_monthly_spending(monthly_total: pd.Series, months_ahead: int = 1):
    if len(monthly_total) < 2:
        return None, None, None

    monthly_total = monthly_total.sort_index()
    X = np.arange(len(monthly_total)).reshape(-1, 1)
    y = monthly_total.values

    reg = LinearRegression()
    reg.fit(X, y)

    future_indices = np.arange(len(monthly_total),
                               len(monthly_total) + months_ahead).reshape(-1, 1)
    preds = reg.predict(future_indices)

    all_indices = np.arange(len(monthly_total) + months_ahead).reshape(-1, 1)
    all_preds = reg.predict(all_indices)

    return preds, all_indices.flatten(), all_preds


# ==============================
# ML: ANOMALY DETECTION
# ==============================

def detect_expense_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    rates = df["rate"]
    if rates.std() == 0:
        df["z_score"] = 0.0
        return df.iloc[0:0]

    df["z_score"] = (df["rate"] - rates.mean()) / rates.std()
    anomalies = df[df["z_score"].abs() >= z_threshold]
    return anomalies


# ==============================
# ML: DAILY CLUSTERING
# ==============================

def cluster_daily_spending(df: pd.DataFrame, n_clusters: int = 3):
    daily = df.pivot_table(
        index="date",
        columns="type",
        values="rate",
        aggfunc="sum",
        fill_value=0
    )

    if daily.shape[0] < n_clusters:
        return None, None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    daily["cluster"] = kmeans.fit_predict(daily)

    daily["total_spent"] = daily.drop(columns=["cluster"]).sum(axis=1)
    cluster_summary = daily.groupby("cluster")["total_spent"].mean()

    return daily, cluster_summary


# ==============================
# SUGGESTIONS ENGINE
# ==============================

def generate_suggestions(df, aggs, forecast_preds, anomalies_df, cluster_summary):
    suggestions = []
    total_spent = df["rate"].sum()
    monthly_total = aggs["monthly_total"]

    # 1) Food share
    food_total = df[df["type"].str.lower().str.contains("food")]["rate"].sum()
    if total_spent > 0:
        food_share = food_total / total_spent
        if food_share > 0.4:
            suggestions.append(
                f"Food-related spending is about {food_share*100:.1f}% of total. "
                f"Reducing snacks/tea outside a few days per week could noticeably lower monthly expenses."
            )

    # 2) Forecast vs average
    if forecast_preds is not None:
        next_month_forecast = forecast_preds[0]
        avg_monthly = monthly_total.mean()
        if next_month_forecast > avg_monthly * 1.1:
            suggestions.append(
                f"Next month is forecasted around {next_month_forecast:.0f}, which is higher than your "
                f"average monthly spending of ~{avg_monthly:.0f}. Consider planning ahead for large recurring "
                f"payments or reducing non-essential costs."
            )
        else:
            suggestions.append(
                f"Next month’s forecast (~{next_month_forecast:.0f}) is close to or below your average "
                f"monthly spending (~{avg_monthly:.0f}). Current spending pattern looks relatively stable."
            )

    # 3) Anomaly-based suggestion
    if anomalies_df is not None and not anomalies_df.empty:
        high_types = anomalies_df["type"].value_counts().index.tolist()
        suggestions.append(
            "There are a few high-value or rare transactions, mainly in: "
            + ", ".join(high_types[:3]) + ". "
            "These are good to mark as planned big expenses (like home support or investments) so "
            "they don’t feel like sudden shocks."
        )

    # 4) Cluster-based suggestion
    if cluster_summary is not None and not cluster_summary.empty:
        max_cluster = cluster_summary.idxmax()
        suggestions.append(
            f"Cluster {max_cluster} has the highest average daily spend (~{cluster_summary[max_cluster]:.0f}). "
            "Check what typically happens on these days (e.g., outings, shopping, or big bills) and limit "
            "how often they occur each month."
        )

    # 5) Investment encouragement
    inv_df = df[df["type"].str.lower().str.contains("investment")]
    if not inv_df.empty:
        inv_total = inv_df["rate"].sum()
        suggestions.append(
            f"You are already investing a total of ~{inv_total:.0f} across the period. "
            "Maintaining consistent monthly investments, even small, is a strong long-term habit."
        )

    if not suggestions:
        suggestions.append("Spending pattern looks quite balanced. You can continue tracking and refining over time.")

    return suggestions


# ==============================
# STREAMLIT APP
# ==============================

def main():
    st.set_page_config(page_title="Expense Dashboard", layout="wide")
    st.title("📊 Expense Analytics & ML Dashboard")

    st.markdown(
        "This dashboard visualizes expenses from `expenses.xlsx` and uses simple ML logic "
        "to highlight patterns, anomalies, and future trends."
    )

    df = load_expense_data("Expense.xlsx")
    aggs = compute_basic_aggregations(df)

    # Sidebar filters (optional)
    st.sidebar.header("Filters")
    years = sorted(df["year"].unique())
    selected_years = st.sidebar.multiselect("Year(s)", years, default=years)
    if selected_years:
        df_filtered = df[df["year"].isin(selected_years)]
    else:
        df_filtered = df.copy()

    # Recompute aggregations on filtered data
    aggs_filtered = compute_basic_aggregations(df_filtered)

    # Tabs for charts
    tab_overview, tab_category, tab_food, tab_modes, tab_charity_inv, tab_ml = st.tabs(
        ["Overview", "By Category", "Food Trend", "Payment Modes", "Charity & Investment", "ML Insights"]
    )

    # ---------------- Overview Tab ----------------
    with tab_overview:
        st.subheader("Monthly Total Spending")
        monthly_total = aggs_filtered["monthly_total"]

        if monthly_total.empty:
            st.info("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(monthly_total.index, monthly_total.values)
            ax.set_title("Monthly Total Spending")
            ax.set_xlabel("Month")
            ax.set_ylabel("Amount")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.write("Total spent in selected period:", int(monthly_total.sum()))

    # ---------------- Category Tab ----------------
    with tab_category:
        st.subheader("Spending by Category")
        category_total = aggs_filtered["category_total"]

        if category_total.empty:
            st.info("No category data available.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(category_total.index.astype(str), category_total.values)
            ax.set_title("Total Spending by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Amount")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.dataframe(category_total.rename("Total Amount"))

    # ---------------- Food Trend Tab ----------------
    with tab_food:
        st.subheader("Daily Food Spending Trend")
        food_daily = aggs_filtered["food_daily"]

        if food_daily.empty:
            st.info("No 'Food' type entries found for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(food_daily.index, food_daily.values, marker="o")
            ax.set_title("Daily Food Spending")
            ax.set_xlabel("Date")
            ax.set_ylabel("Amount")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # ---------------- Payment Modes Tab ----------------
    with tab_modes:
        st.subheader("Spending by Mode of Transaction")
        mode_total = aggs_filtered["mode_total"]

        if mode_total.empty:
            st.info("No mode of transaction data available.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(mode_total.index.astype(str), mode_total.values)
            ax.set_title("Total Spending by Payment Mode")
            ax.set_xlabel("Mode")
            ax.set_ylabel("Amount")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # ---------------- Charity & Investment Tab ----------------
    with tab_charity_inv:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Charity by Month")
            charity_by_month = aggs_filtered["charity_by_month"]
            if charity_by_month.empty:
                st.info("No charity records for the selected filters.")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(charity_by_month.index, charity_by_month.values)
                ax.set_title("Charity Amount by Month")
                ax.set_xlabel("Month")
                ax.set_ylabel("Amount")
                plt.xticks(rotation=45)
                st.pyplot(fig)

        with col2:
            st.subheader("Investment by Month")
            inv_by_month = aggs_filtered["inv_by_month"]
            if inv_by_month.empty:
                st.info("No investment records for the selected filters.")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(inv_by_month.index, inv_by_month.values)
                ax.set_title("Investment Amount by Month")
                ax.set_xlabel("Month")
                ax.set_ylabel("Amount")
                plt.xticks(rotation=45)
                st.pyplot(fig)

        st.subheader("Essential vs Non-Essential Spending")
        ess_split = aggs_filtered["ess_split"]
        if ess_split.empty:
            st.info("No essential/non-essential split available.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(ess_split.index, ess_split.values)
            ax.set_title("Essential vs Non-Essential")
            ax.set_ylabel("Amount")
            st.pyplot(fig)

    # ---------------- ML Insights Tab ----------------
    with tab_ml:
        st.subheader("ML Insights & Suggestions")

        # 1) Category suggestions (demo)
        model = train_category_suggestion_model(df_filtered)
        demo_results = get_category_suggestions_demo(model)

        with st.expander("Smart Category Suggestions (Demo)"):
            if demo_results is None:
                st.info("Not enough labeled data to train a category suggestion model.")
            else:
                for comment, pred in demo_results:
                    st.write(f"**{comment}** → Suggested type: `{pred}`")

        # 2) Forecast chart
        st.markdown("### Monthly Spending Forecast")
        monthly_total = aggs_filtered["monthly_total"]
        forecast_preds, all_indices, all_preds = forecast_monthly_spending(monthly_total, months_ahead=1)

        if forecast_preds is None:
            st.info("Not enough monthly data to generate a forecast.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            X_hist = np.arange(len(monthly_total))
            ax.scatter(X_hist, monthly_total.values, label="Actual")
            ax.plot(all_indices, all_preds, label="Trend & Forecast")
            ax.set_xlabel("Month Index (0 = first month in filtered data)")
            ax.set_ylabel("Total Spending")
            ax.legend()
            st.pyplot(fig)

            st.write(f"**Forecast for next month:** ~ {forecast_preds[0]:.0f}")

        # 3) Anomaly detection
        st.markdown("### Anomalous / Unusual Expenses")
        anomalies_df = detect_expense_anomalies(df_filtered, z_threshold=2.0)
        if anomalies_df.empty:
            st.info("No strong anomalies detected with the current threshold.")
        else:
            st.dataframe(
                anomalies_df[["date", "type", "comments", "rate", "z_score"]]
                .sort_values("z_score", ascending=False)
            )

        # 4) Daily clustering
        st.markdown("### Daily Spending Clusters")
        clustered_daily, cluster_summary = cluster_daily_spending(df_filtered, n_clusters=3)
        if clustered_daily is None:
            st.info("Not enough days to perform clustering.")
        else:
            # Show counts
            cluster_counts = clustered_daily["cluster"].value_counts().sort_index()
            st.write("Number of days in each cluster:")
            st.dataframe(cluster_counts.rename("day_count"))

            # Bar chart of avg spend per cluster
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(cluster_summary.index.astype(str), cluster_summary.values)
            ax.set_title("Average Daily Spending per Cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Average Spend")
            st.pyplot(fig)

        # 5) Suggestions under the ML charts
        st.markdown("### 💡 Suggestions")
        suggestions = generate_suggestions(
            df_filtered,
            aggs_filtered,
            forecast_preds,
            anomalies_df if not anomalies_df.empty else None,
            cluster_summary if clustered_daily is not None else None,
        )
        for s in suggestions:
            st.markdown(f"- {s}")


if __name__ == "__main__":
    main()
