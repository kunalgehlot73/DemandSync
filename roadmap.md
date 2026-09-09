# Project Roadmap

## Phase 0 — Freeze the project definition

### Final title

**Machine Learning-Based Demand Forecasting for Inventory Optimization in Retail Supply Chains**

### Main research/engineering question

> Can machine-learning-based demand forecasts improve inventory replenishment decisions and reduce total inventory cost compared with conventional forecasting and inventory policies?

### Core hypothesis

> More accurate and appropriately calibrated demand forecasts should enable better replenishment decisions, leading to lower total inventory cost and/or higher service levels.

---

# Phase 1 — Understand SAPiBench completely

Before writing any ML code, inspect the dataset.

### Step 1. Download the repository

Obtain:

```text
material_master.csv
synthetic_7year_demand.csv
synthetic_generator.py
SAPiBench.ipynb
README.md
```

The repository already provides an end-to-end notebook and generator, which we'll use primarily for **understanding and validating the benchmark**, not blindly copying its workflow. ([GitHub][1])

### Step 2. Understand every column

Create a data dictionary:

| Variable         | Meaning               | Type        | Used for     |
| ---------------- | --------------------- | ----------- | ------------ |
| Material ID      | SKU identifier        | Categorical | Model        |
| Category         | Product category      | Categorical | Model        |
| Demand           | Daily demand          | Numeric     | Target       |
| Base demand      | Typical demand        | Numeric     | Feature      |
| Seasonality      | Seasonal behavior     | Categorical | Feature      |
| Lead time        | Supplier delay        | Numeric     | Inventory    |
| Unit cost        | Product cost          | Numeric     | Inventory    |
| Holding cost     | Cost of holding stock | Numeric     | Optimization |
| Ordering cost    | Cost per order        | Numeric     | Optimization |
| Stockout penalty | Shortage cost         | Numeric     | Optimization |

The exact column names should be confirmed directly from the files before implementation.

### Step 3. Understand how the synthetic data was generated

This matters because synthetic data has known underlying patterns.

SAPiBench generates demand using monthly and weekly seasonality, category-specific effects, item-specific base demand, and stochastic noise. ([GitHub][1])

We need to document these characteristics rather than accidentally treating them as unknown real-world processes.

---

# Phase 2 — Exploratory Data Analysis

Now understand the demand behavior.

## Step 4. Analyze demand distribution

Check:

* Mean demand
* Median
* Standard deviation
* Minimum/maximum
* Skewness
* Zero-demand frequency
* Coefficient of variation

Do this overall and by SKU/category.

---

## Step 5. Analyze time patterns

Plot:

```text
Daily demand
Weekly demand
Monthly demand
Yearly demand
```

Look for:

* Trends
* Weekly seasonality
* Monthly seasonality
* Annual seasonality
* Demand spikes
* Low-demand products

---

## Step 6. Analyze SKU heterogeneity

Not every product behaves the same way.

Cluster or classify products into groups such as:

```text
High demand / low variability
High demand / high variability
Low demand / low variability
Low demand / high variability
```

This will become important during model evaluation.

---

# Phase 3 — Data Preparation

## Step 7. Create the forecasting table

The fundamental structure should become something like:

```text
Date
SKU
Category
Demand
Lag_1
Lag_2
Lag_7
Lag_14
Lag_28
Rolling_Mean_7
Rolling_Mean_28
Rolling_STD_28
DayOfWeek
Month
...
```

---

## Step 8. Feature engineering

We'll create four major feature groups.

### A. Historical demand

```text
lag_1
lag_2
lag_3
lag_7
lag_14
lag_28
lag_56
```

### B. Rolling statistics

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
rolling_std_7
rolling_std_28
EWMA
```

### C. Calendar features

```text
day_of_week
week_of_year
month
quarter
weekend
```

### D. Static/item features

```text
category
base_demand
seasonality
lead_time
unit_cost
holding_cost
ordering_cost
stockout_penalty
```

But we'll only include inventory variables in the forecasting model where there's a defensible reason they would be known at prediction time.

---

# Phase 4 — Leakage Prevention

This is one of the most important parts.

We **cannot** randomly shuffle the dataset.

Instead:

```text
2019 ───────────── 2023 | 2024 | 2025
       TRAIN             VAL      TEST
```

The exact date boundary will be selected after examining the data.

For every forecast:

> The model can only see information that would have been available at that point in time.

For example, the model predicting demand for January 10 cannot use January 11 demand indirectly through rolling statistics or preprocessing.

---

# Phase 5 — Establish Baselines

Before sophisticated ML, establish simple benchmarks.

## Model 0 — Naive

```text
Forecast tomorrow = today's demand
```

## Model 1 — Seasonal Naive

```text
Forecast Monday = previous Monday demand
```

## Model 2 — Moving Average

```text
Forecast = mean(previous 7 days)
```

These baselines are essential. A sophisticated model isn't useful unless it can outperform simple alternatives.

---

# Phase 6 — XGBoost Model

This becomes our principal tabular ML model.

## Step 9. Build training features

Input:

```text
Lag features
Rolling statistics
Calendar information
SKU/category information
```

Output:

```text
7-day demand forecast
```

---

## Step 10. Hyperparameter optimization

Use **Optuna** to search parameters such as:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
reg_alpha
reg_lambda
```

Optimization objective:

> Minimize validation WAPE/MASE, with secondary consideration of RMSE.

---

# Phase 7 — LSTM

Now implement sequence forecasting.

## Step 11. Create sequences

For example:

```text
Previous 28 days
        ↓
     LSTM
        ↓
Next 7 days
```

The model sees sequences rather than manually engineered lag columns alone.

Architecture can initially be:

```text
Input
 ↓
LSTM
 ↓
LSTM
 ↓
Dense
 ↓
7 outputs
```

Hyperparameters to tune:

* Sequence length
* Hidden dimensions
* Number of layers
* Dropout
* Learning rate
* Batch size

---

# Phase 8 — Temporal Fusion Transformer

This is the advanced forecasting model.

## Step 12. Build TFT

TFT is suitable for this problem because we have:

* Static SKU information
* Historical demand
* Time-varying features
* Known calendar variables
* Multi-step forecasting

Architecture conceptually:

```text
Static features
       │
Historical demand ──► TFT ──► 7-day forecast
       │
Calendar variables
```

Unlike XGBoost, TFT can also provide useful information about temporal feature importance and variable selection.

---

# Phase 9 — Probabilistic Forecasting

This is a significant upgrade over simply predicting a point estimate.

Inventory decisions care about uncertainty.

Instead of:

> Expected demand = 150

we want something like:

```text
P10 = 120
P50 = 150
P90 = 195
```

This lets the inventory system reason about demand uncertainty.

TFT is particularly useful here because we can configure it for **quantile forecasting**.

---

# Phase 10 — Forecast Evaluation

Evaluate:

### Primary

**WAPE**

### Secondary

**MAE**

**RMSE**

**MASE**

For each model:

```text
Naive
Seasonal Naive
Moving Average
XGBoost
LSTM
TFT
```

Create:

| Model          | MAE | RMSE | WAPE | MASE |
| -------------- | --: | ---: | ---: | ---: |
| Naive          |     |      |      |      |
| Seasonal Naive |     |      |      |      |
| Moving Average |     |      |      |      |
| XGBoost        |     |      |      |      |
| LSTM           |     |      |      |      |
| TFT            |     |      |      |      |

Also evaluate separately by:

* SKU
* Category
* Demand volatility
* Forecast horizon

---

# Phase 11 — Forecast Uncertainty Evaluation

For probabilistic predictions, evaluate:

* Prediction Interval Coverage Probability
* Mean Interval Width
* Quantile loss / Pinball loss

For example:

> Does the 90% prediction interval actually contain approximately 90% of observed demand?

This becomes especially valuable for inventory optimization.

---

# Phase 12 — Build the Inventory Simulator

Now we move from **prediction** to **decision-making**.

This is the heart of the project.

The simulator tracks:

```text
Inventory
Demand
Orders
Lead time
Receipts
Stockouts
Holding cost
Ordering cost
Shortage cost
```

SAPiBench explicitly supports simulation of reorder behavior, purchase-order timing, stockouts, ending inventory, and total cost. ([GitHub][1])

---

# Phase 13 — Conventional Inventory Policy

Implement the standard approach first.

## Safety Stock

Use demand uncertainty and lead time.

Conceptually:

```text
Safety Stock = service-level factor × demand uncertainty × lead-time factor
```

## Reorder Point

```text
ROP = expected lead-time demand + safety stock
```

## Order quantity

Implement:

> **EOQ**

as a benchmark.

---

# Phase 14 — Forecast-Driven Inventory Policy

Now replace historical averages with ML forecasts.

Instead of:

```text
Historical average
       ↓
ROP
       ↓
Order
```

we have:

```text
ML forecast
       ↓
Expected lead-time demand
       ↓
Forecast uncertainty
       ↓
Safety stock
       ↓
ROP
       ↓
EOQ/order quantity
```

Run this separately for XGBoost, LSTM and TFT.

---

# Phase 15 — Cost-Based Optimization

Now make inventory optimization genuinely optimization-oriented.

Objective:

$$
\min \; C_{holding}+C_{ordering}+C_{stockout}
$$

subject to an appropriate service-level requirement.

We'll optimize parameters such as:

```text
Reorder point
Safety-stock multiplier
Order-up-to level
```

using the validation period.

**Never optimize them on the test set.**

---

# Phase 16 — Compare Inventory Policies

We'll run at least:

### Policy 1

Historical-demand baseline

### Policy 2

Seasonal-naive forecast + standard inventory policy

### Policy 3

XGBoost + standard inventory policy

### Policy 4

LSTM + standard inventory policy

### Policy 5

TFT + standard inventory policy

### Policy 6

Best forecast + optimized inventory policy

This gives us a strong experimental matrix.

---

# Phase 17 — Inventory Evaluation

For every policy calculate:

### Cost

```text
Total Cost
Holding Cost
Ordering Cost
Stockout Cost
```

### Operational

```text
Service Level
Fill Rate
Stockout Days
Average Inventory
Inventory Turnover
Number of Orders
```

The objective isn't necessarily to minimize one metric independently.

For example:

> A strategy that reduces inventory cost by 30% but creates unacceptable stockouts isn't automatically better.

---

# Phase 18 — Forecast Accuracy vs Inventory Performance

This is arguably the most interesting analysis.

Create a comparison such as:

```text
             Forecast Error
                   │
                   ▼
             Inventory Cost
                   │
                   ▼
              Service Level
```

Then examine:

> Does a lower forecast error actually produce lower total inventory cost?

This prevents the project from assuming that forecasting accuracy and business performance are identical.

---

# Phase 19 — Ablation Study

Remove components one at a time.

For example:

### Experiment A

XGBoost with basic features

### Experiment B

XGBoost + lag features

### Experiment C

XGBoost + lag + rolling features

### Experiment D

XGBoost + all temporal features

This identifies **which information actually contributes to performance**.

We can perform a similar analysis for the inventory system.

---

# Phase 20 — Sensitivity Analysis

Change important business conditions.

For example:

### Service levels

```text
90%
95%
98%
99%
```

### Lead times

```text
Short
Medium
Long
```

### Stockout penalty

```text
Low
Medium
High
```

Then observe how optimal inventory policies change.

This produces much more meaningful results than a single experiment.

---

# Phase 21 — Robustness Testing

Test models on:

* High-demand products
* Low-demand products
* Highly volatile products
* Strongly seasonal products
* Different forecast horizons

We should identify **where the models work and where they fail**.

---

# Phase 22 — Statistical Comparison

Once results are available, don't just say:

> TFT is better.

Test whether the improvement is statistically meaningful.

For example, use appropriate paired tests across forecast origins/SKUs to compare forecast errors or policy costs.

This strengthens the experimental claims.

---

# Phase 23 — Visualization

The final project should have strong visual outputs.

### Forecast plots

```text
Actual vs predicted demand
```

### Model comparison

```text
MAE / RMSE / WAPE / MASE
```

### Inventory plots

```text
Inventory level over time
Reorder points
Purchase orders
Stockouts
```

### Cost breakdown

```text
Holding
Ordering
Stockout
Total
```

### Business trade-off

For example:

```text
Service Level ↑
       vs
Total Cost ↑
```

This will make the results much easier to interpret.

---

# Phase 24 — Build the Final System

Once experimentation is complete, create a clean application/pipeline.

### Input

```text
SKU
Historical demand
Inventory state
Lead time
Costs
```

### Processing

```text
Feature engineering
       ↓
TFT/XGBoost/etc.
       ↓
Demand forecast
       ↓
Uncertainty estimation
       ↓
Inventory optimization
```

### Output

```text
Predicted demand
Recommended safety stock
Reorder point
Recommended order quantity
Expected inventory
Expected cost
Stockout risk
```

---

# Phase 25 — Dashboard

A dashboard would be a useful final layer, but **only after the ML experiments are finished**.

Possible tools:

**Streamlit** for the main application.

Example:

```text
SKU: MAT_023

Predicted 7-day demand: 1,240 units
Current inventory:       650 units
Safety stock:            280 units
Reorder point:           540 units

Recommended order:       750 units
Expected service level:  97.2%
Expected total cost:     ₹XXXX
```

The currency isn't important for the synthetic benchmark; we can report cost in the dataset's monetary units.

---

# Phase 26 — Reproducibility

Create a clean project structure:

```text
inventory-optimization/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_lstm.ipynb
│   ├── 05_tft.ipynb
│   └── 06_inventory_optimization.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── forecasting/
│   ├── inventory/
│   ├── simulation/
│   └── evaluation/
│
├── configs/
│
├── results/
│
├── app/
│
├── requirements.txt
│
└── README.md
```

Keep raw data immutable and save model configurations, seeds, and experiment results.

---

# Phase 27 — Final experiment

The final experiment should look like this:

```text
                     SAPiBench
                         │
                         ▼
                 Train / Validation
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        XGBoost         LSTM           TFT
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Demand Forecasts
                         │
                         ▼
                Inventory Simulator
                         │
                         ▼
             Cost-Based Optimization
                         │
                         ▼
                Held-Out Test Set
                         │
                         ▼
      ┌──────────────────────────────────┐
      │ Forecast Metrics                 │
      │ Inventory Cost                   │
      │ Service Level                    │
      │ Stockouts                        │
      │ Average Inventory                │
      └──────────────────────────────────┘
```

---

# Phase 28 — Research conclusions

At the end, we should be able to answer **five concrete questions**:

### Q1

Which forecasting model predicts demand most accurately?

### Q2

Does forecast accuracy differ by SKU/category/demand volatility?

### Q3

Does better forecasting actually reduce inventory cost?

### Q4

Which inventory policy performs best under different cost/service-level conditions?

### Q5

Is the added complexity of advanced models such as TFT justified by the resulting inventory benefits?

Those questions give the project a much stronger analytical foundation.

---

# Final Development Order

To avoid getting overwhelmed, **we should actually build it in this order**:

**1. Dataset acquisition and inspection**
↓
**2. Data dictionary**
↓
**3. EDA**
↓
**4. Preprocessing**
↓
**5. Time-series splitting**
↓
**6. Naive baselines**
↓
**7. XGBoost**
↓
**8. LSTM**
↓
**9. TFT**
↓
**10. Forecast evaluation**
↓
**11. Uncertainty/quantile evaluation**
↓
**12. Inventory simulator**
↓
**13. ROP + Safety Stock + EOQ baseline**
↓
**14. Forecast-driven inventory policies**
↓
**15. Cost-based optimization**
↓
**16. Inventory evaluation**
↓
**17. Forecast-vs-inventory analysis**
↓
**18. Ablation + sensitivity + robustness experiments**
↓
**19. Final model/policy selection**
↓
**20. Dashboard/application**
↓
**21. Documentation and research paper**


[1]: https://github.com/clcamar74/SAPiBench-inventory-simulation-dataset-2026?utm_source=chatgpt.com "GitHub - clcamar74/SAPiBench-inventory-simulation-dataset-2026: SAPiBench inventory simulation dataset 2026 · GitHub"
