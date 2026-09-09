# SAPiBench Data Dictionary

Confirmed directly from the generated CSVs (`data/raw/`), 2026-09-09.
Two tables, joined on `Material_ID`.

## `material_master.csv` (50 rows — one per SKU)

| Column              | Type        | Meaning                                   | Used for                          |
|---------------------|-------------|--------------------------------------------|------------------------------------|
| Material_ID          | Categorical | SKU identifier (MAT001–MAT050)             | Join key / model feature           |
| Category             | Categorical | 9 product categories (FMCG, Grocery, ...)  | Static feature                     |
| Description          | Text        | Human-readable item name                   | Not modeled                        |
| Base_Daily_Demand    | Numeric     | Typical daily demand baseline (5–60)       | Redundant w/ demand table, feature |
| Seasonality_Type     | Categorical | None / Summer / Holiday / School / Rainy   | Static feature — **see gotcha #1** |
| Lead_Time            | Numeric     | Supplier lead time in days (3, 5, 7, or 10)| Inventory policy input             |
| Unit_Cost            | Numeric     | Cost per unit (50–500)                     | Inventory optimization             |
| Holding_Cost_Rate    | Numeric     | Holding cost as a rate (0.01–0.05)         | Inventory optimization             |
| Ordering_Cost        | Numeric     | Fixed cost per order (200–800)             | Inventory optimization (EOQ)       |
| Stockout_Penalty     | Numeric     | Shortage cost per unit short (100–500)     | Inventory optimization             |

## `synthetic_7year_demand.csv` (127,850 rows = 50 SKUs × 2,557 days)

| Column           | Type     | Meaning                                          | Used for               |
|------------------|----------|---------------------------------------------------|-------------------------|
| Date             | Date     | 2019-01-01 to 2025-12-31, daily, no gaps           | Time index              |
| Material_ID      | Categorical | SKU identifier, join key to material_master     | Join key                |
| Base_Demand      | Numeric  | Copy of Base_Daily_Demand (redundant column)       | Drop or ignore           |
| Seasonal_Factor  | Numeric  | Monthly/category seasonal multiplier applied       | **Leakage risk — see gotcha #2** |
| Weekly_Factor    | Numeric  | Day-of-week multiplier applied                     | **Leakage risk — see gotcha #2** |
| Noise            | Numeric  | The exact stochastic noise term used (±20%)        | **Leakage risk — see gotcha #2** |
| Final_Demand     | Numeric  | Realized daily demand (target variable)            | **Target**               |

Note: costs (holding/ordering/stockout) and lead time live only in
`material_master.csv`, not in the demand table — any inventory-simulation
step needs a join on `Material_ID`, not a lookup within the demand table
itself.

## Gotchas found during inspection

1. **`Seasonality_Type` string "None" reads as NaN by default.** The
   generator writes the literal string `"None"` for 40/50 items. Pandas'
   default `read_csv` treats the string `"None"` as a null value, silently
   turning 80% of this column into `NaN`. Anyone loading this with default
   settings and later doing `fillna()` or dropping nulls could accidentally
   destroy the meaning "no special seasonality" instead of encoding it.
   Fix: always load material_master with `keep_default_na=False`, or
   explicitly recode `"None"` to a sentinel like `"none"` right after load.
   `generate_dataset.py` doesn't yet special-case this — worth fixing at
   generation time too, since the same trap awaits every downstream user.

2. **`Seasonal_Factor`, `Weekly_Factor`, and `Noise` are generation
   artifacts, not real-world observable features — and `Noise` in
   particular is the exact random draw used to produce `Final_Demand`
   for that row.** If any of these three columns are used as model
   features, the model isn't forecasting demand, it's reverse-engineering
   the noise term that *created* demand — a severe (and very easy to miss)
   form of target leakage, distinct from the temporal leakage Phase 4
   already plans to guard against. **Decision: drop `Seasonal_Factor`,
   `Weekly_Factor`, and `Noise` before any modeling.** They're useful only
   for validating that the generator's seasonality logic matches what we
   see in Phase 2 EDA — nothing else.

## Discrepancy vs. source repo's README

The README claims a "100-item Walmart-style synthetic material master."
The actual `material_catalog` in `synthetic_generator.py` has exactly 50
items (MAT001–MAT050), confirmed by both the source and our regenerated
CSV. Documenting this now so it doesn't cause confusion later (e.g. if
someone assumes 100 SKUs are available for the SKU-heterogeneity
clustering in Phase 2, Step 6).

## Zero-demand / intermittency

Min `Final_Demand` = 3, **0.00% zero-demand rows** across all 127,850
rows. This dataset has no intermittent-demand SKUs — every item sells
something every day. That's a real limitation vs. typical retail data
(where slow movers often show 20-60% zero-demand days) and means
intermittent-demand-specific techniques (Croston's method, TSB, zero-
inflated models) aren't needed here, but also that any conclusions about
"the model handles low-demand products well" won't generalize to genuinely
intermittent series.
