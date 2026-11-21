"""
Data-backed SWOT.

Each point in the SWOT pulls its number from the analysis, so if the data changes the
SWOT text changes with it. The wording is mine; the numbers are computed.
"""
import pandas as pd

WEAK_SCORE = 3.2          # an aspect scoring below this is treated as a weakness


def weak_aspects(reviews: pd.DataFrame, fuel_by_model: dict) -> pd.DataFrame:
    r = reviews.copy()
    r["fuel_type"] = r["model"].map(fuel_by_model)
    return r[(r["score"] < WEAK_SCORE) & (r["fuel_type"] == "EV")].sort_values("score")


def build_swot(ctx: dict) -> dict:
    ov, seg, mix, perf, growth, pot, weak = (ctx[k] for k in
        ["overview", "ev_segment", "ev_model_mix", "model_perf", "growth", "potential", "weak"])
    y = int(ov.index.max())
    first_ev_year = int(ov.index.min())
    share_now = ov.loc[y, "company_share_of_ev_segment_pct"]
    share_peak = ov["company_share_of_ev_segment_pct"].max()
    rival = seg.drop(columns="Company X").loc[y].idxmax()
    rival_share = seg.loc[y, rival]
    fastest = seg.drop(columns="Company X").diff().loc[y].idxmax()
    fastest_gain = seg.drop(columns="Company X").diff().loc[y].max()
    city = perf.loc["X-City EV"]
    base = pot[(pot["ev_adoption"] == "base") & pot["company_share_scenario"].str.startswith("defend")].iloc[0]
    trend = pot[(pot["ev_adoption"] == "base") & pot["company_share_scenario"].str.startswith("trend")].iloc[0]

    return {
        "Strengths": [
            f"Clear EV leader: {share_now:.1f}% of the country's EV segment in {y} "
            f"(next brand, {rival}, has {rival_share:.1f}%).",
            f"First mover: selling EVs since {first_ev_year}, with a 3-model EV range covering compact, mid and premium SUVs.",
            f"EVs are now {ov.loc[y, 'ev_pct_of_company_sales']:.1f}% of company sales, up from "
            f"{ov.loc[first_ev_year, 'ev_pct_of_company_sales']:.1f}%, so the business is already shifting.",
            f"Strong petrol/CNG base ({ov.loc[y, 'company_market_share_pct']:.1f}% of the total market) funds the EV transition.",
        ],
        "Weaknesses": [
            f"EV segment share has fallen from {share_peak:.1f}% to {share_now:.1f}%.",
            f"Best-selling EV (X-City EV) is losing weight in the mix: {mix.loc[y, 'X-City EV']:.0f}% of company EV "
            f"sales in {y}, down from {mix['X-City EV'].max():.0f}%; it grew {city['yoy_pct']:.0f}% YoY while the "
            f"EV market grew {growth.loc['EV', 'last_yoy_pct']:.0f}%.",
        ] + [f"Low customer score for {r.model} on {r.aspect.replace('_', ' ')}: {r.score}/5."
             for r in weak.itertuples()],
        "Opportunities": [
            f"EV sales are growing {growth.loc['EV', 'cagr_pct']:.0f}% a year (CAGR) while petrol grows "
            f"{growth.loc['Petrol', 'cagr_pct']:.0f}%.",
            f"Base scenario: the EV market reaches {base['ev_market_units']:,} units in 2030. Holding today's share "
            f"means {base['company_ev_units']:,} units (~US$ {base['revenue_busd']} bn), "
            f"{base['growth_vs_now_x']}x today's EV volume.",
            f"X-City Petrol is the company's best seller ({perf.loc['X-City Petrol', 'units']:,.0f} units in {y}); "
            "its buyers are a natural upgrade path to X-City EV, which shares the name and body style.",
        ],
        "Threats": [
            f"{fastest} gained {fastest_gain:.1f} pp of the EV segment in {y} alone.",
            f"If the current trend continues, share falls to {trend['company_ev_share_pct']}% by 2030: "
            f"{base['company_ev_units'] - trend['company_ev_units']:,} fewer units than holding share (base scenario).",
            f"EV adoption speed is uncertain: the 2030 EV share ranges from {ctx['forecast'].query('year == 2030')['ev_share_pct'].min():.0f}% "
            f"to {ctx['forecast'].query('year == 2030')['ev_share_pct'].max():.0f}% across scenarios, so capacity plans need flexibility.",
        ],
    }


def recommendations(ctx: dict) -> list:
    weak = ctx["weak"]
    recs = []
    city_weak = weak[weak["model"] == "X-City EV"]
    if len(city_weak):
        aspects = " and ".join(a.replace("_", " ") for a in city_weak["aspect"])
        recs.append(f"**Upgrade X-City EV ({aspects})**: it is the volume EV and the entry point to the "
                    "brand. A long-range battery option and faster DC charging target its two lowest scores "
                    "without changing its price advantage (price score is its best aspect).")
    pro = weak[weak["model"] == "X-Pro EV"]
    if len(pro):
        recs.append("**Add a lower trim of X-Pro EV**: its only weak score is price, and it is the fastest "
                    "growing EV model, so a cheaper variant could widen its reach.")
    pot, fc = ctx["potential"], ctx["forecast"]
    base = pot[pot["ev_adoption"] == "base"].set_index("company_share_scenario")["company_ev_units"]
    gap = base.max() - base.min()
    recs.append(f"**Track EV segment share monthly against the fastest-growing rival**: in the base scenario, "
                f"defending share vs letting the trend continue is worth {gap:,.0f} units in 2030 alone.")
    p30 = fc[fc["year"] == fc["year"].max()]["petrol_share_pct"]
    recs.append(f"**Keep petrol/CNG models but cap new investment**: petrol is still {p30.min():.0f}-{p30.max():.0f}% "
                f"of the market in {fc['year'].max()} depending on the scenario, so it funds the EV plan, "
                "but its share falls in every scenario.")
    return recs
