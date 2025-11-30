import pandas as pd
import matplotlib.pyplot as plt

# === 1. LOAD & CLEAN DATA ===

# Change this to your actual file name
file_path = "Expense.xlsx"

df = pd.read_excel(file_path)

# Normalize column names (in case of spaces/case issues)
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Expecting columns like:
# date, type, comments, rate, mode_of_transaction, monthly_total

# Drop pre-calculated monthly total rows (we'll recalc ourselves)
if "monthly_total" in df.columns:
    df = df[df["monthly_total"].isna()]

# Parse date column
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])

# Ensure rate is numeric
df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0)

# Add month & year-month for grouping
df["month"] = df["date"].dt.to_period("M").astype(str)
df["year"] = df["date"].dt.year

# Fill missing mode_of_transaction with "Unknown"
if "mode_of_transaction" in df.columns:
    df["mode_of_transaction"] = df["mode_of_transaction"].fillna("Unknown")
else:
    df["mode_of_transaction"] = "Unknown"

# === 2. MONTHLY TOTAL SPENDING (BAR CHART) ===

monthly_total = df.groupby("month")["rate"].sum().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(monthly_total.index, monthly_total.values)
plt.title("Monthly Total Spending")
plt.xlabel("Month")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 3. CATEGORY-WISE SPENDING (BAR CHART) ===

category_total = df.groupby("type")["rate"].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(category_total.index.astype(str), category_total.values)
plt.title("Total Spending by Category")
plt.xlabel("Category")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 4. FOOD EXPENSE TREND OVER TIME (LINE CHART) ===

food_df = df[df["type"].str.lower().str.contains("food")]

if not food_df.empty:
    food_daily = food_df.groupby("date")["rate"].sum()

    plt.figure(figsize=(10, 5))
    plt.plot(food_daily.index, food_daily.values, marker="o")
    plt.title("Daily Food Spending Trend")
    plt.xlabel("Date")
    plt.ylabel("Amount Spent on Food")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# === 5. PAYMENT MODE DISTRIBUTION (BAR CHART) ===

mode_total = df.groupby("mode_of_transaction")["rate"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 4))
plt.bar(mode_total.index.astype(str), mode_total.values)
plt.title("Total Spending by Mode of Transaction")
plt.xlabel("Payment Mode")
plt.ylabel("Total Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 6. CHARITY & INVESTMENTS OVER TIME (SEPARATE CHARTS) ===

# Charity trend
charity_df = df[df["type"].str.lower().str.contains("charity")]

if not charity_df.empty:
    charity_by_month = charity_df.groupby("month")["rate"].sum().sort_index()

    plt.figure(figsize=(8, 4))
    plt.bar(charity_by_month.index, charity_by_month.values)
    plt.title("Charity Amount by Month")
    plt.xlabel("Month")
    plt.ylabel("Charity Total")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Investment trend
inv_df = df[df["type"].str.lower().str.contains("investment")]

if not inv_df.empty:
    inv_by_month = inv_df.groupby("month")["rate"].sum().sort_index()

    plt.figure(figsize=(8, 4))
    plt.bar(inv_by_month.index, inv_by_month.values)
    plt.title("Investment Amount by Month")
    plt.xlabel("Month")
    plt.ylabel("Investment Total")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# === 7. ESSENTIAL vs NON-ESSENTIAL SPLIT (BAR CHART) ===

# You can tweak these based on your preferences
essentials = ["food", "travel", "home", "health", "clothing", "recharge", "loan"]
df["is_essential"] = df["type"].str.lower().apply(
    lambda t: "Essential" if any(e in t for e in essentials) else "Non-Essential"
)

ess_split = df.groupby("is_essential")["rate"].sum()

plt.figure(figsize=(6, 4))
plt.bar(ess_split.index, ess_split.values)
plt.title("Essential vs Non-Essential Spending")
plt.xlabel("Type")
plt.ylabel("Total Amount")
plt.tight_layout()
plt.show()
