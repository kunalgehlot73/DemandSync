"""
Regenerates the SAPiBench synthetic dataset locally.

This is the source repo's synthetic_generator.py, byte-for-byte identical
in logic (same seed=42, same date range, same catalog, same seasonality
rules), with only the output paths changed from Google Drive to our local
data/raw/ folder. See data/raw/synthetic_generator_ORIGINAL.py for the
untouched original, pulled from:
https://github.com/clcamar74/SAPiBench-inventory-simulation-dataset-2026

Run:
    python src/data/generate_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# -----------------------------
# PARAMETERS
# -----------------------------
start_date = "2019-01-01"
end_date = "2025-12-31"

# -----------------------------
# 50-ITEM WALMART-STYLE CATALOG
# -----------------------------
material_catalog = [
    ("MAT001", "FMCG", "Shampoo", 35, "None"),
    ("MAT002", "FMCG", "Conditioner", 30, "None"),
    ("MAT003", "FMCG", "Toothpaste", 40, "None"),
    ("MAT004", "FMCG", "Bath Soap", 45, "None"),
    ("MAT005", "FMCG", "Laundry Detergent", 50, "None"),
    ("MAT006", "FMCG", "Fabric Softener", 30, "None"),
    ("MAT007", "FMCG", "Dishwashing Liquid", 25, "None"),
    ("MAT008", "FMCG", "Bottled Water 1L", 60, "None"),
    ("MAT009", "FMCG", "Instant Coffee", 35, "None"),
    ("MAT010", "FMCG", "Canned Tuna", 28, "None"),
    ("MAT011", "Grocery", "Rice 5kg", 40, "None"),
    ("MAT012", "Grocery", "Cooking Oil 1L", 30, "None"),
    ("MAT013", "Grocery", "Sugar 1kg", 25, "None"),
    ("MAT014", "Grocery", "Flour 1kg", 20, "None"),
    ("MAT015", "Grocery", "Pasta 500g", 18, "None"),
    ("MAT016", "Grocery", "Tomato Sauce 250g", 22, "None"),
    ("MAT017", "Grocery", "Breakfast Cereal", 15, "None"),
    ("MAT018", "Grocery", "Peanut Butter", 12, "None"),
    ("MAT019", "Grocery", "Chocolate Bar", 30, "None"),
    ("MAT020", "Grocery", "Snack Chips", 35, "None"),
    ("MAT021", "Perishable", "Fresh Milk 1L", 50, "None"),
    ("MAT022", "Perishable", "Eggs 12pcs", 45, "None"),
    ("MAT023", "Perishable", "Bread Loaf", 40, "None"),
    ("MAT024", "Perishable", "Bananas", 38, "None"),
    ("MAT025", "Perishable", "Chicken Breast 1kg", 30, "None"),
    ("MAT026", "Household", "Toilet Paper 12pk", 25, "None"),
    ("MAT027", "Household", "Paper Towels", 20, "None"),
    ("MAT028", "Household", "Trash Bags", 18, "None"),
    ("MAT029", "Household", "Light Bulbs", 12, "None"),
    ("MAT030", "Household", "Cleaning Spray", 15, "None"),
    ("MAT031", "Electronics", "Earphones", 12, "None"),
    ("MAT032", "Electronics", "USB Cable", 15, "None"),
    ("MAT033", "Electronics", "Power Bank", 10, "None"),
    ("MAT034", "Electronics", "Wireless Mouse", 8, "None"),
    ("MAT035", "Electronics", "LED Flashlight", 10, "None"),
    ("MAT036", "Apparel", "Men T-Shirt", 20, "Summer"),
    ("MAT037", "Apparel", "Women T-Shirt", 18, "Summer"),
    ("MAT038", "Apparel", "Socks 3pk", 15, "None"),
    ("MAT039", "Apparel", "Baseball Cap", 10, "Summer"),
    ("MAT040", "Apparel", "Slippers", 12, "Summer"),
    ("MAT041", "Seasonal", "Christmas Lights", 5, "Holiday"),
    ("MAT042", "Seasonal", "Gift Wrap", 8, "Holiday"),
    ("MAT043", "Seasonal", "School Notebook", 20, "School"),
    ("MAT044", "Seasonal", "Ballpoint Pens", 25, "School"),
    ("MAT045", "Seasonal", "Electric Fan", 10, "Summer"),
    ("MAT046", "Toys", "Toy Car", 8, "None"),
    ("MAT047", "Toys", "Stuffed Animal", 6, "None"),
    ("MAT048", "Toys", "Coloring Book", 10, "None"),
    ("MAT049", "Misc", "Water Bottle", 15, "None"),
    ("MAT050", "Misc", "Umbrella", 12, "Rainy"),
]


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)

    material_master = pd.DataFrame(
        material_catalog,
        columns=["Material_ID", "Category", "Description", "Base_Daily_Demand", "Seasonality_Type"],
    )

    material_master["Lead_Time"] = np.random.choice([3, 5, 7, 10], len(material_master))
    material_master["Unit_Cost"] = np.random.uniform(50, 500, len(material_master)).round(2)
    material_master["Holding_Cost_Rate"] = np.random.uniform(0.01, 0.05, len(material_master)).round(3)
    material_master["Ordering_Cost"] = np.random.uniform(200, 800, len(material_master)).round(2)
    material_master["Stockout_Penalty"] = np.random.uniform(100, 500, len(material_master)).round(2)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    monthly_factor = {
        1: 0.9, 2: 0.95, 3: 1.0, 4: 1.05, 5: 1.1, 6: 1.15,
        7: 1.1, 8: 1.0, 9: 0.95, 10: 1.0, 11: 1.2, 12: 1.5,
    }
    weekly_factor = {0: 1.05, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.05, 5: 1.1, 6: 1.1}

    rows = []
    for _, mat in material_master.iterrows():
        for date in dates:
            base = mat["Base_Daily_Demand"]
            month = date.month
            weekday = date.weekday()

            season_mult = monthly_factor[month]
            if mat["Seasonality_Type"] == "Holiday" and month == 12:
                season_mult *= 2.0
            if mat["Seasonality_Type"] == "School" and month in [6, 7]:
                season_mult *= 1.5
            if mat["Seasonality_Type"] == "Summer" and month in [3, 4, 5]:
                season_mult *= 1.3
            if mat["Seasonality_Type"] == "Rainy" and month in [6, 7, 8, 9]:
                season_mult *= 1.3

            week_mult = weekly_factor[weekday]
            noise = np.random.uniform(-0.2, 0.2)
            final_demand = max(0, int(base * season_mult * week_mult * (1 + noise)))

            rows.append([date, mat["Material_ID"], base, season_mult, week_mult, noise, final_demand])

    synthetic_data = pd.DataFrame(
        rows,
        columns=["Date", "Material_ID", "Base_Demand", "Seasonal_Factor", "Weekly_Factor", "Noise", "Final_Demand"],
    )

    return material_master, synthetic_data


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    material_master, synthetic_data = generate()
    material_master.to_csv(RAW_DIR / "material_master.csv", index=False)
    synthetic_data.to_csv(RAW_DIR / "synthetic_7year_demand.csv", index=False)
    print(f"Saved to {RAW_DIR}:")
    print(" - material_master.csv")
    print(" - synthetic_7year_demand.csv")


if __name__ == "__main__":
    main()
