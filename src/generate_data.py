"""
generate_data.py
----------------
Generates a realistic synthetic dataset of personal financial transactions
for 500 users over 12 months and saves to data/transactions_synthetic.csv
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
NUM_USERS       = 500
MONTHS          = 12
TARGET_MIN_ROWS = 6_000
TARGET_MAX_ROWS = 10_000

END_DATE   = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=365)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "transactions_synthetic.csv"
)

# ── Merchant catalogue per category ─────────────────────────────────────────
MERCHANTS = {
    "Food":           ["Zomato", "Swiggy", "Domino's", "McDonald's", "Subway",
                       "Blinkit", "BigBasket", "Dunzo", "Haldiram's", "Chai Point"],
    "Transport":      ["Uber", "Ola", "Rapido", "IRCTC", "IndiGo Airlines",
                       "RedBus", "MakeMyTrip", "Metro Card", "Petrol Pump", "Yulu"],
    "Entertainment":  ["BookMyShow", "PVR Cinemas", "INOX", "Disney+ Hotstar",
                       "YouTube Premium", "Steam", "Gaana", "JioSaavn", "Apple TV+"],
    "Subscriptions":  ["Netflix", "Amazon Prime", "Spotify", "Gym Membership",
                       "Jio Fiber", "Airtel Xstream", "Zee5", "SonyLIV", "MX Player"],
    "Utilities":      ["BESCOM", "TATA Power", "Indane Gas", "Airtel Postpaid",
                       "Jio Recharge", "BSNL", "Water Bill", "Society Maintenance"],
    "Shopping":       ["Amazon", "Flipkart", "Myntra", "Nykaa", "Meesho",
                       "IKEA", "Croma", "Reliance Digital", "DMart", "Ajio"],
    "Medical":        ["Apollo Pharmacy", "1mg", "Netmeds", "Practo", "PharmEasy",
                       "Max Healthcare", "Manipal Hospital", "Lenskart", "HealthifyMe"],
    "Salary":         ["Employer Salary Credit"],
}

SUBSCRIPTION_SERVICES = [
    ("Netflix",          599),
    ("Amazon Prime",     299),
    ("Spotify",          119),
    ("Disney+ Hotstar",  299),
    ("Gym Membership",   800),
    ("Jio Fiber",        699),
    ("Airtel Xstream",   349),
    ("Zee5",             99),
    ("SonyLIV",          299),
    ("YouTube Premium",  189),
]

# ── Helper: random date within a given month ──────────────────────────────────
def random_date_in_month(year: int, month: int, weekday_bias: bool = False) -> datetime:
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1)
    end   = datetime(year, month, last_day)

    if weekday_bias:
        weekends = [start + timedelta(days=d)
                    for d in range((end - start).days + 1)
                    if (start + timedelta(days=d)).weekday() >= 4]
        if weekends:
            return random.choice(weekends)

    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def clamp_date(dt: datetime) -> datetime:
    return max(START_DATE, min(END_DATE, dt))

# ── Month list ────────────────────────────────────────────────────────────────
def month_range():
    months = []
    cur = START_DATE.replace(day=1)
    while cur <= END_DATE:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months

MONTHS_LIST = month_range()

# ── Build user profiles ───────────────────────────────────────────────────────
def build_user_profiles(num_users: int) -> list:
    profiles = []
    for i in range(num_users):
        user_id = f"USR{i+1:04d}"
        salary  = random.randint(30_000, 80_000)
        num_subs = random.choices([1, 2, 3, 4], weights=[20, 40, 30, 10])[0]
        subs = random.sample(SUBSCRIPTION_SERVICES, k=num_subs)
        low_usage_mask = [random.random() < 0.3 for _ in subs]
        profiles.append({
            "user_id":       user_id,
            "salary":        salary,
            "subscriptions": subs,
            "low_usage":     low_usage_mask,
        })
    return profiles

# ── Transaction generators ────────────────────────────────────────────────────

def make_row(user_id, date, amount, category, merchant, txn_type):
    return {
        "user_id":          user_id,
        "date":             date.strftime("%Y-%m-%d"),
        "amount":           round(amount, 2),
        "category":         category,
        "merchant_name":    merchant,
        "transaction_type": txn_type,
        "description":      f"{merchant} - {category}",
    }

def generate_salary_transactions(profile):
    rows = []
    uid = profile["user_id"]
    for year, month in MONTHS_LIST:
        salary_date = clamp_date(datetime(year, month, 1))
        rows.append(make_row(uid, salary_date, profile["salary"],
                             "Salary", "Employer Salary Credit", "Credit"))
    return rows

def generate_subscription_transactions(profile):
    rows = []
    uid = profile["user_id"]
    for (service, base_price), _ in zip(profile["subscriptions"], profile["low_usage"]):
        for year, month in MONTHS_LIST:
            bill_day = max(1, min(28, 5 + random.randint(-3, 3)))
            try:
                bill_date = clamp_date(datetime(year, month, bill_day))
            except ValueError:
                bill_date = clamp_date(datetime(year, month, 5))
            amount = max(49, base_price + random.randint(-20, 20))
            rows.append(make_row(uid, bill_date, amount, "Subscriptions", service, "Debit"))
    return rows

def generate_essential_transactions(profile):
    rows = []
    uid  = profile["user_id"]
    for year, month in MONTHS_LIST:
        # Utilities
        for _ in range(random.randint(1, 2)):
            rows.append(make_row(uid,
                                 clamp_date(random_date_in_month(year, month)),
                                 random.uniform(200, 3_000),
                                 "Utilities", random.choice(MERCHANTS["Utilities"]), "Debit"))
        # Transport
        for _ in range(random.randint(3, 8)):
            rows.append(make_row(uid,
                                 clamp_date(random_date_in_month(year, month)),
                                 random.uniform(50, 1_200),
                                 "Transport", random.choice(MERCHANTS["Transport"]), "Debit"))
        # Food
        for _ in range(random.randint(4, 10)):
            rows.append(make_row(uid,
                                 clamp_date(random_date_in_month(year, month)),
                                 random.uniform(80, 2_500),
                                 "Food", random.choice(MERCHANTS["Food"]), "Debit"))
    return rows

def generate_weekend_impulse_transactions(profile):
    rows = []
    uid  = profile["user_id"]
    impulse_cats = ["Entertainment", "Shopping", "Food"]
    for year, month in MONTHS_LIST:
        for _ in range(random.randint(1, 5)):
            cat = random.choice(impulse_cats)
            rows.append(make_row(uid,
                                 clamp_date(random_date_in_month(year, month, weekday_bias=True)),
                                 random.uniform(300, 8_000),
                                 cat, random.choice(MERCHANTS[cat]), "Debit"))
    return rows

def generate_medical_transactions(profile):
    rows = []
    uid  = profile["user_id"]
    for year, month in MONTHS_LIST:
        if random.random() < 0.25:
            rows.append(make_row(uid,
                                 clamp_date(random_date_in_month(year, month)),
                                 random.uniform(100, 5_000),
                                 "Medical", random.choice(MERCHANTS["Medical"]), "Debit"))
    return rows

def generate_shopping_transactions(profile):
    rows = []
    uid  = profile["user_id"]
    for year, month in MONTHS_LIST:
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                rows.append(make_row(uid,
                                     clamp_date(random_date_in_month(year, month)),
                                     random.uniform(200, 15_000),
                                     "Shopping", random.choice(MERCHANTS["Shopping"]), "Debit"))
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Financial Wellness Engine — Synthetic Data Generator")
    print("=" * 60)
    print(f"  Users       : {NUM_USERS}")
    print(f"  Period      : {START_DATE.date()} -> {END_DATE.date()}")
    print(f"  Target rows : {TARGET_MIN_ROWS:,} - {TARGET_MAX_ROWS:,}")
    print("=" * 60)

    profiles  = build_user_profiles(NUM_USERS)
    all_rows  = []

    print("\n  Generating transactions...")
    for profile in profiles:
        all_rows.extend(generate_salary_transactions(profile))
        all_rows.extend(generate_subscription_transactions(profile))
        all_rows.extend(generate_essential_transactions(profile))
        all_rows.extend(generate_weekend_impulse_transactions(profile))
        all_rows.extend(generate_medical_transactions(profile))
        all_rows.extend(generate_shopping_transactions(profile))

    df = pd.DataFrame(all_rows)
    df["date"]   = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].round(2)
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Trim to TARGET_MAX_ROWS if exceeded
    if len(df) > TARGET_MAX_ROWS:
        df = (df.sample(n=TARGET_MAX_ROWS, random_state=SEED)
                .sort_values(["user_id", "date"])
                .reset_index(drop=True))
    elif len(df) < TARGET_MIN_ROWS:
        extra_needed = TARGET_MIN_ROWS - len(df)
        extras = df.sample(n=extra_needed, replace=True, random_state=SEED).copy()
        extras["amount"] = (extras["amount"] *
                            np.random.uniform(0.9, 1.1, size=len(extras))).round(2)
        df = (pd.concat([df, extras], ignore_index=True)
                .sort_values(["user_id", "date"])
                .reset_index(drop=True))

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  CSV saved -> {OUTPUT_PATH}")
    print("\n" + "-" * 60)
    print(f"  Total rows generated  : {len(df):,}")
    print(f"  Date range            : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Amount range (INR)    : Rs {df['amount'].min():,.2f} -> Rs {df['amount'].max():,.2f}")
    print(f"  Unique users          : {df['user_id'].nunique()}")
    print(f"  Transaction types     : {df['transaction_type'].value_counts().to_dict()}")

    print("\n  Category Distribution:")
    cat_dist = df["category"].value_counts()
    for cat, count in cat_dist.items():
        pct = count / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"    {cat:<15} {count:>6,}  ({pct:5.1f}%)  {bar}")

    print("\n  Sample 5 rows:")
    print(df.sample(5, random_state=SEED).to_string(index=False))
    print("\n" + "=" * 60)
    print("  Data generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
