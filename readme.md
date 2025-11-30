# 📊 Expense Visualizer (Python)

This project visualizes personal expense data stored in an Excel file using Python.  
It helps understand spending patterns through clear charts and trends.

---

## 📌 Overview

The visualizer script generates multiple graphs automatically, including:

- Monthly total spending
- Spending by category (Food, Travel, Charity, Home, etc.)
- Food expense trend over time
- Payment mode distribution (Cash / UPI / Card)
- Charity and investment spending by month
- Essential vs Non-Essential spending comparison

These insights help track financial discipline and identify spending improvements.

---

## 📂 Project Structure

```

project/
├─ expenses.xlsx
├─ expense_visualiser.py
└─ README.md

```

---

## 📑 Excel Input Format (`expenses.xlsx`)

The file must contain the following columns:

| Column Name             | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| **Date**                | Transaction date (DD-MM-YYYY format)                     |
| **Type**                | Expense category (e.g., Food, Travel, Charity, Clothing) |
| **Comments**            | Optional description of the expense                      |
| **Rate**                | Amount spent                                             |
| **Mode of Transaction** | Cash / UPI / Card / Other (optional)                     |
| **Monthly Total**       | _(Optional)_ ignored by script if present                |

➡️ The script automatically removes rows with pre-calculated monthly totals.

---

## 🛠 Setup & Installation

### Requirements

- Python **3.13.9** (or any version ≥ 3.8)
- Windows / macOS / Linux

---

### Running the project

1️⃣ **Create a virtual environment (PowerShell)**

```ps
python -m venv testenv
```

2️⃣ **Activate virtual environment**

```ps
testenv\Scripts\Activate.ps1
```

3️⃣ **Install dependencies**

```ps
pip install pandas matplotlib openpyxl
```

4️⃣ **Run the script**

```ps
python expense_visualiser.py
```

---

## 📈 Charts Generated

| Chart                      | Description                               |
| -------------------------- | ----------------------------------------- |
| Monthly Total Spending     | Shows spending trend across months        |
| Category-wise Spending     | Highlights where most money is spent      |
| Food Daily Trend           | Tracks food expenses day by day           |
| Payment Mode Summary       | Cash vs UPI vs Card spending distribution |
| Charity Trend              | Monthly charity contribution tracking     |
| Investment Trend           | Monthly capital allocation tracking       |
| Essential vs Non-Essential | Spending discipline comparison            |

Each chart will be displayed automatically via **matplotlib**.

---

## 🔒 Data Security

- The script **only reads** the Excel file — no overwriting or editing occurs.
- All data remains **local to your machine** and is **not shared externally**.

---

## 🚀 Future Enhancements (Optional)

Possible add-ons:

- Save charts as PNG automatically
- Export monthly report as PDF
- Streamlit / Web Dashboard for interactive filtering
- Savings calculator based on income vs expense

If you want any of these upgrades, they can be added easily.

---

## 💡 Contribution

Feel free to modify or extend the charts based on your requirements.
Pull requests and improvements are welcome.

---

```

```
