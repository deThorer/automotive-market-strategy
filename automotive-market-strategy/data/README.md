# Data

All three files are **simulated** by `scripts/generate_case_data.py` (default seed 3).
To use real data, keep the same columns.

### market_sales.csv (whole country)
| column | description |
|---|---|
| month | `YYYY-MM` |
| brand | "Company X", "Competitor A".. "Others" |
| fuel_type | EV, Petrol, Diesel, CNG, Hybrid |
| units | new cars registered |

### company_models.csv (Company X only)
| column | description |
|---|---|
| month | `YYYY-MM` |
| model | e.g. X-City EV |
| segment | Compact SUV, Mid SUV, Premium SUV |
| fuel_type | must match market_sales |
| units | units sold (must add up to Company X totals in market_sales, checked in `src/data.py`) |
| avg_price_usd | average transaction price |

### model_reviews.csv
| column | description |
|---|---|
| model | Company X model |
| aspect | range, charging, price, build_quality, features, mileage |
| score | average customer score, 1-5 |
| n_reviews | number of reviews |

How the simulation works (short version): total market grows ~5%/yr with an Oct-Nov
festive peak and an Apr-Jun 2020 lockdown dip. EV share follows the early part of a
logistic curve. Company X's EV segment share is 88% until mid-2020 and then erodes as
Competitor A (2020) and Competitor C (2023) enter. Random noise is added to every series.
