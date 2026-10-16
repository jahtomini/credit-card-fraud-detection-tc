"""Generate synthetic IEEE-CIS-style transaction and identity CSVs for demoing the app.

The output follows the raw Kaggle schema (train_transaction.csv / train_identity.csv)
so it can be uploaded straight into the Streamlit app. Values are synthetic: they
resemble the real distributions loosely, with fraud rows skewed towards known risk
signals, but they are not real data and are not suitable for evaluating the model.

Usage:
    python generate_sample_data.py --rows 1000 --seed 17
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

START_TRANSACTION_ID = 2_987_000
START_TRANSACTION_DT = 86_400  # The dataset's TransactionDT starts at one day.

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com",
    "comcast.net", "icloud.com", "outlook.com", "msn.com", "att.net",
    "live.com", "sbcglobal.net", "verizon.net", "ymail.com", "bellsouth.net",
]
RISKY_EMAIL_DOMAINS = ["outlook.com", "mail.com", "protonmail.com", "anonymous.com", "gmail.com"]
CARD_BRANDS = ["visa", "mastercard", "american express", "discover"]
CARD5_VALUES = [226.0, 224.0, 166.0, 102.0, 117.0, 138.0, 195.0, 137.0, 126.0, 219.0]
OS_VALUES = ["Windows 10", "Windows 7", "iOS 11.2.1", "iOS 11.1.2", "Android 7.0",
             "Mac OS X 10_12_6", "Mac OS X 10_11_6", "Android 8.0.0", "Linux"]
BROWSERS = ["chrome 63.0", "mobile safari 11.0", "chrome 64.0", "mobile safari generic",
            "ie 11.0 for desktop", "safari generic", "edge 16.0", "firefox 57.0",
            "chrome 63.0 for android", "samsung browser 6.2"]
SCREENS = ["1920x1080", "1366x768", "2208x1242", "1334x750", "1440x900",
           "2436x1125", "1600x900", "2560x1440", "1280x800"]
DESKTOP_INFO = ["Windows", "MacOS", "Trident/7.0", "rv:57.0", "Linux"]
MOBILE_INFO = ["iOS Device", "SM-G930V Build/NRD90M", "SM-J700M Build/MMB29K",
               "Moto G (4) Build/NPJ25.93-14.7", "SAMSUNG SM-G532M Build/MMB29T",
               "LG-H870 Build/NRD90U"]

TRANSACTION_COLUMNS = (
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"M{i}" for i in range(1, 10)]
    + [f"V{i}" for i in range(1, 340)]
)
IDENTITY_COLUMNS = (
    ["TransactionID"]
    + [f"id_{i:02d}" for i in range(1, 39)]
    + ["DeviceType", "DeviceInfo"]
)


def maybe(rng: random.Random, value, missing_rate: float):
    """Return value, or None (an empty CSV cell) with the given probability."""
    return None if rng.random() < missing_rate else value


def transaction_row(rng: random.Random, tx_id: int, dt: int, is_fraud: bool) -> dict:
    product = rng.choices(["W", "C", "R", "H", "S"],
                          weights=[20, 50, 10, 12, 8] if is_fraud else [75, 10, 7, 6, 2])[0]

    if is_fraud:
        amount = rng.lognormvariate(4.8, 1.1)
    elif product == "W":
        amount = rng.lognormvariate(4.3, 0.9)
    else:
        amount = rng.lognormvariate(3.6, 0.8)
    # Product C transactions in the source data are often in non-USD cents.
    amount = round(min(amount, 5000.0), 3 if product == "C" else 2)

    card1 = rng.randint(1000, 18396)
    addr1 = maybe(rng, float(rng.randint(100, 540)), 0.35 if is_fraud else 0.1)
    email = rng.choice(RISKY_EMAIL_DOMAINS if is_fraud and rng.random() < 0.6 else EMAIL_DOMAINS)
    has_ident_email = product != "W"

    row = {
        "TransactionID": tx_id,
        "isFraud": int(is_fraud),
        "TransactionDT": dt,
        "TransactionAmt": amount,
        "ProductCD": product,
        "card1": card1,
        "card2": maybe(rng, float(rng.randint(100, 600)), 0.02),
        "card3": 150.0 if product == "W" or rng.random() < 0.6 else float(rng.choice([185, 119, 144])),
        "card4": rng.choices(CARD_BRANDS, weights=[65, 32, 2, 1])[0],
        "card5": maybe(rng, rng.choice(CARD5_VALUES), 0.01),
        "card6": rng.choices(["debit", "credit"],
                             weights=[45, 55] if is_fraud else [75, 25])[0],
        "addr1": addr1,
        "addr2": None if addr1 is None else rng.choices([87.0, 60.0, 96.0], weights=[95, 3, 2])[0],
        "dist1": maybe(rng, float(int(rng.expovariate(1 / 120))), 0.6),
        "dist2": maybe(rng, float(int(rng.expovariate(1 / 200))), 0.93),
        "P_emaildomain": maybe(rng, email, 0.15),
        "R_emaildomain": email if has_ident_email and rng.random() < 0.7 else None,
    }

    burst = 6 if is_fraud else 1
    for i in range(1, 15):
        row[f"C{i}"] = float(min(int(rng.expovariate(1 / (1.5 * burst))), 3000))

    days_since = 0.0 if is_fraud and rng.random() < 0.5 else float(rng.randint(0, 640))
    for i in range(1, 16):
        missing = 0.93 if i == 7 else (0.02 if i == 1 else 0.45)
        value = days_since if i in (1, 2) else float(rng.randint(0, 640))
        row[f"D{i}"] = maybe(rng, value, missing)

    m_missing = 0.75 if is_fraud else (0.15 if product == "W" else 0.85)
    for i in range(1, 10):
        if i == 4:
            row["M4"] = maybe(rng, rng.choice(["M0", "M1", "M2"]), m_missing)
        else:
            row[f"M{i}"] = maybe(rng, "T" if rng.random() < 0.7 else "F", m_missing)

    # V features: Vesta-engineered counts and amount aggregates, missing in blocks.
    block_missing = {block: rng.random() < 0.4 for block in range(12)}
    for i in range(1, 340):
        if block_missing[i * 12 // 340]:
            row[f"V{i}"] = None
        elif 126 <= i <= 137 or 202 <= i <= 216 or 263 <= i <= 278 or 306 <= i <= 321:
            row[f"V{i}"] = round(amount * rng.randint(0, 3 * burst), 3) if rng.random() < 0.3 else 0.0
        else:
            row[f"V{i}"] = float(rng.choices([0, 1, 2, 3], weights=[30, 60, 7, 3])[0])

    return row


def identity_row(rng: random.Random, tx_id: int, is_fraud: bool) -> dict:
    mobile = rng.random() < (0.55 if is_fraud else 0.35)
    os_name = rng.choice([o for o in OS_VALUES if (o.startswith(("iOS", "Android")) == mobile)])
    browser = rng.choice([b for b in BROWSERS if (("mobile" in b or "android" in b
                                                   or "samsung" in b) == mobile)] or BROWSERS)
    found = "NotFound" if is_fraud and rng.random() < 0.5 else "Found"

    row = {
        "TransactionID": tx_id,
        "id_01": float(-5 * rng.randint(0, 20)),
        "id_02": maybe(rng, float(rng.randint(1, 999_595)), 0.02),
        "id_03": maybe(rng, 0.0, 0.55),
        "id_04": maybe(rng, 0.0, 0.55),
        "id_05": maybe(rng, float(rng.randint(0, 5)), 0.05),
        "id_06": maybe(rng, float(-rng.randint(0, 10)), 0.05),
        "id_07": maybe(rng, float(rng.randint(-20, 40)), 0.96),
        "id_08": maybe(rng, float(rng.randint(-100, 0)), 0.96),
        "id_09": maybe(rng, 0.0, 0.5),
        "id_10": maybe(rng, 0.0, 0.5),
        "id_11": maybe(rng, 100.0, 0.02),
        "id_12": found,
        "id_13": maybe(rng, float(rng.choice([49, 52, 33, 64, 20])), 0.1),
        "id_14": maybe(rng, float(rng.choice([-300, -360, -480, -420, 60])), 0.45),
        "id_15": rng.choice(["New", "Found", "Unknown"]) if is_fraud else rng.choice(["Found", "Found", "New"]),
        "id_16": found,
        "id_17": float(rng.choice([166, 225, 100, 102])),
        "id_18": maybe(rng, float(rng.choice([15, 13, 12])), 0.92),
        "id_19": float(rng.randint(100, 671)),
        "id_20": float(rng.randint(100, 661)),
        "id_28": "New" if is_fraud and rng.random() < 0.6 else rng.choice(["Found", "New"]),
        "id_29": found,
        "id_30": maybe(rng, os_name, 0.45),
        "id_31": browser,
        "id_32": maybe(rng, float(rng.choice([24, 32])), 0.45),
        "id_33": maybe(rng, rng.choice(SCREENS), 0.5),
        "id_34": maybe(rng, rng.choice(["match_status:2", "match_status:1"]), 0.45),
        "id_35": rng.choice(["T", "F"]),
        "id_36": rng.choice(["F", "F", "F", "T"]),
        "id_37": rng.choice(["T", "T", "F"]),
        "id_38": rng.choice(["T", "F"]),
        "DeviceType": "mobile" if mobile else "desktop",
        "DeviceInfo": maybe(rng, rng.choice(MOBILE_INFO if mobile else DESKTOP_INFO), 0.2),
    }
    # id_21..id_27 were dropped in training (>90% missing); keep them almost always empty.
    for i in range(21, 28):
        row[f"id_{i}"] = None if rng.random() < 0.96 else float(rng.randint(0, 900))
    return row


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=1000, help="number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.035,
                        help="share of fraudulent rows (IEEE-CIS is about 3.5%%)")
    parser.add_argument("--identity-coverage", type=float, default=0.25,
                        help="base share of transactions with an identity row (fraud rows get more)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-label", action="store_true", help="omit the isFraud column")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    transactions, identities = [], []
    dt = START_TRANSACTION_DT
    for n in range(args.rows):
        tx_id = START_TRANSACTION_ID + n
        dt += max(1, int(rng.expovariate(1 / 30)))
        is_fraud = rng.random() < args.fraud_rate
        tx = transaction_row(rng, tx_id, dt, is_fraud)
        transactions.append(tx)

        # Identity rows mainly exist for non-W products, as in the source data.
        coverage = args.identity_coverage * (0.2 if tx["ProductCD"] == "W" else 3.0)
        if is_fraud:
            coverage *= 1.5
        if rng.random() < min(coverage, 0.95):
            identities.append(identity_row(rng, tx_id, is_fraud))

    tx_columns = [c for c in TRANSACTION_COLUMNS if not (args.no_label and c == "isFraud")]
    tx_path = args.out_dir / "sample_transactions.csv"
    id_path = args.out_dir / "sample_identity.csv"
    write_csv(tx_path, tx_columns, transactions)
    write_csv(id_path, IDENTITY_COLUMNS, identities)

    frauds = sum(t["isFraud"] for t in transactions)
    print(f"Wrote {tx_path} ({len(transactions):,} rows, {frauds} labelled fraud, "
          f"{len(tx_columns)} columns)")
    print(f"Wrote {id_path} ({len(identities):,} rows, {len(IDENTITY_COLUMNS)} columns)")


if __name__ == "__main__":
    main()
