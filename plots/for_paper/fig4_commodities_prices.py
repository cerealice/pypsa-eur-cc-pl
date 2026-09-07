# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:58:40 2025

@author: Dibella
"""

# %% IMPORTS AND CONFIG

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np

# === CONFIGURATION ===
scenarios = [
    #"base_reg_deindustrial",
    #"base_reg_regain",
    "policy_reg_deindustrial",
    "policy_reg_maintain",
    "policy_reg_regain"
]
years = [2020, 2030, 2040, 2050]


scenario_colors = {
    "base_reg_deindustrial": "#813746",
    "policy_reg_deindustrial": "#FC814A",
    "base_reg_regain": "#6D8088",
    "policy_reg_regain": "#28C76F",
    "policy_reg_maintain": "#deca4b",
}

scenario_labels = {
    #"base_reg_deindustrial": "No Climate Policy\nDeindustrialization",
    "policy_reg_deindustrial": "Climate Policy\nContinued Decline",
    "policy_reg_maintain": "Climate Policy\nStabilization",
    #"base_reg_regain": "No Climate Policy\nReindustrialization",
    "policy_reg_regain": "Climate Policy\nReindustrialization"
}

lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t
naphtha_to_hvc = (2.31 * 12.47) * 1000
decay_emis_hvc = 0.2571 * naphtha_to_hvc / 1e3
lhv_hydrogen = 33.33  # MWh/t

# === HISTORICAL VALUES ===
hist_2020_prices = {
    "steel": 420, #https://www.spglobal.com/commodity-insights/en/news-research/latest-news/metals/030121-europe-hrc-steel-to-raw-materials-spread-rises-on-steel-prices
    "cement": 93, #https://pubs.usgs.gov/periodicals/mcs2024/mcs2024-cement.pdf
    "ammonia": 470, # https://www.chemanalyst.com/Pricing-data/ammonia-37
    "methanol": 400, # https://www.sciencedirect.com/science/article/pii/S2666792421000421
    "HVC": 600, #https://www.iea.org/data-and-statistics/charts/simplified-levelised-cost-of-petrochemicals-for-selected-feedstocks-and-regions-2017?utm_source=chatgpt.com
    #"H2": 1800
}

ft_to_hvc = 2.31 * 12.47
max_import_costs = {
    "steel": 501,
    "ammonia": 145.8 * lhv_ammonia,
    "methanol": 168.5 * lhv_methanol,
    "HVC": 182.4 * ft_to_hvc,
    #"H2": 132.1 * lhv_hydrogen
}

min_import_costs = {
    "steel": 417,
    "ammonia": 87.7 * lhv_ammonia,
    "methanol": 106.8 * lhv_methanol,
    "HVC": 109.8 * ft_to_hvc,
    #"H2": 57.5 * lhv_hydrogen
}


def weighted_average_marginal_price(n, keyword, exclude_labels=None):
    mprice_cols = n.buses_t.marginal_price.columns[
        n.buses_t.marginal_price.columns.str.contains(keyword)
    ]
    if exclude_labels:
        for label in exclude_labels:
            mprice_cols = mprice_cols[~mprice_cols.str.contains(label)]
    mprice = n.buses_t.marginal_price.loc[:, mprice_cols].where(lambda df: df >= 0, 0)
    relevant_loads = mprice.columns.intersection(n.loads.index)
    mprice_loads = mprice[relevant_loads]
    loads_w_mprice = n.loads_t.p[relevant_loads]

    if keyword == "H2":
        loads_links = n.links[n.links.bus1.str.endswith(' H2') & ~n.links.index.str.contains('pipeline')]
        loads = -n.links_t.p1.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads_w_mprice = loads.T.groupby(level=0).sum().T
        mprice.columns = mprice.columns.str[:2]
        mprice_loads = mprice.T.groupby(level=0).sum().T
    elif keyword == 'methanol':
        loads_links = n.loads[n.loads.index.str.endswith('methanol')]
        loads = n.loads_t.p.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads_w_mprice = loads.T.groupby(level=0).sum().T
        mprice.columns = mprice.columns.str[:2]
        mprice_loads = mprice.T.groupby(level=0).sum().T

    total_costs = (mprice_loads * loads_w_mprice).sum().sum()
    weighted_avg = total_costs / loads_w_mprice.sum().sum()
    return weighted_avg


# === INIT STORAGE ===
price_data = {commodity: pd.DataFrame(index=scenarios, columns=years) for commodity in hist_2020_prices.keys()}

for commodity, val in hist_2020_prices.items():
    price_data[commodity][2020] = val


# Extra: for methanol and HVC without CO2 decay cost
price_data["methanol_noco2"] = pd.DataFrame(index=scenarios, columns=years)
price_data["HVC_noco2"] = pd.DataFrame(index=scenarios, columns=years)

price_data["methanol_noco2"][2020] = hist_2020_prices["methanol"]
price_data["HVC_noco2"][2020] = hist_2020_prices["HVC"]

# === LOAD AND COMPUTE PRICES ===
cwd = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(cwd))

for scenario in scenarios:
    for year in years[1:]:
        file_path = os.path.join(parent_dir, "results_april/results", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)

        price_data["steel"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="steel") / 1000  # bus in kt → €/kt → €/t
        price_data["cement"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="cement", exclude_labels=["process emissions"]) 
        price_data["ammonia"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="NH3") * lhv_ammonia
        price_data["methanol"].loc[scenario, year] = weighted_average_marginal_price(n, keyword='industry methanol') * lhv_methanol

        co2_price = -n.global_constraints.loc["CO2Limit", "mu"]
        extra_methanol_cost = 0.2482 * lhv_methanol * co2_price
        extra_hvc_cost = decay_emis_hvc * co2_price  # 0.2571 t_CO2/MWh_naphtha * ft_to_hvc MWh/t_HVC

        price_data["methanol_noco2"].loc[scenario, year] = price_data["methanol"].loc[scenario, year] - extra_methanol_cost
        #price_data["methanol_noco2"].loc[scenario, year] = weighted_average_marginal_price(n, keyword='methanol', exclude_labels=["industry", "shipping"]) * lhv_methanol

        hvc_price = weighted_average_marginal_price(n, keyword="HVC") / 1000  # bus in kt → €/kt → €/t
        price_data["HVC"].loc[scenario, year] = hvc_price
        price_data["HVC_noco2"].loc[scenario, year] = hvc_price - extra_hvc_cost

        #price_data["H2"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="H2", exclude_labels=["pipeline"]) * lhv_hydrogen

price_data["steel"].loc[:, 2030] += 250 # ADB fix capital costs of existing plants
price_data["methanol_noco2"].loc['base_reg_regain',:] = np.nan
price_data["methanol_noco2"].loc['base_reg_deindustrial',:] = np.nan
#   Modelling error
#price_data["methanol_noco2"].loc['policy_reg_regain',2050] += 100
price_data["HVC_noco2"].loc['base_reg_regain',:] = np.nan
price_data["HVC_noco2"].loc['base_reg_deindustrial',:] = np.nan
# %%
# === PLOT ===
commodities = ["steel", "cement", "ammonia", "methanol", "HVC"]#, "H2"]
fig, axes = plt.subplots(1, len(commodities), figsize=(14, 6), sharex=True, sharey=False)

for idx, (commodity, ax) in enumerate(zip(commodities, axes)):
    for scenario in scenarios:
        label = scenario_labels[scenario] if idx == 0 else None
        ax.plot(
            years,
            price_data[commodity].loc[scenario],
            marker="o",
            linestyle="-",
            label=label,
            color=scenario_colors.get(scenario, 'black')
        )

        # Add dashed lines for methanol and HVC without CO2 cost
        if commodity.lower() == "methanol":
            ax.plot(
                years,
                price_data["methanol_noco2"].loc[scenario],
                marker="o",
                linestyle="--",
                color=scenario_colors.get(scenario, 'black'),
                label=None
            )
        elif commodity.upper() == "HVC":
            ax.plot(
                years,
                price_data["HVC_noco2"].loc[scenario],
                marker="o",
                linestyle="--",
                color=scenario_colors.get(scenario, 'black'),
                label=None
            )

    if commodity != "cement":
        if 2050 in years:
            # Draw shaded import price range band from 2045 to 2055
            ax.fill_between(
                [2039, 2051],
                [min_import_costs[commodity]] * 2,
                [max_import_costs[commodity]] * 2,
                color="grey",
                alpha=0.3,
                label="Import price range" if idx == 0 else None
            )
    
            # Horizontal dashed lines for min and max import prices
            ax.hlines(
                y=min_import_costs[commodity],
                xmin=2039, xmax=2051,
                colors='black', linestyles='-.', linewidth=1
            )
            ax.hlines(
                y=max_import_costs[commodity],
                xmin=2039, xmax=2051,
                colors='black', linestyles='-.', linewidth=1
            )
    
    
    custom_titles = {
        "hvc": "Plastics",
        #"h2": "Hydrogen"
    }
    title = custom_titles.get(commodity.lower(), commodity.title())
    ax.set_title(f"{title.capitalize()}")
    ax.set_xticks(years)
    ax.set_ylim(bottom=0)
    if idx == 0:
        ax.set_ylabel("Price [EUR/t]")
        ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linestyle='--')
    

    """
    # Additional legend for line styles
    line_incl = Line2D([0], [0], color='black', linestyle='-', label="Includes EOL CO₂ cost")
    line_excl = Line2D([0], [0], color='black', linestyle='--', label="Excludes EOL CO₂ cost")
    
    fig.legend(
        handles=[line_incl, line_excl],
        loc='lower centger',
        ncol=2,
        bbox_to_anchor=(0.5, 0),
        fontsize=9,
        frameon=False
    )
    """

    if commodity.lower() == "methanol":
        # Add line style legend inside the hydrogen subplot
        ax.legend(
            handles=[
                Line2D([0], [0], color='black', linestyle='-', label="Includes EOL CO₂ cost"),
                Line2D([0], [0], color='black', linestyle='--', label="Excludes EOL CO₂ cost")
            ],
            loc="lower right",
            fontsize=8,
            frameon=True
        )
        
        
plt.tight_layout(rect=[0, 0.15, 1, 1])  # Make space for bottom legend


plt.savefig("./graphs/commodity_prices.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
import matplotlib.pyplot as plt

# === Commodity list ===
commodities = ["steel", "cement", "ammonia", "methanol", "HVC"]

# === Current price references (€/t) with sources ===
prices_europe = {
    "steel": 600,     # €/t – Europe HRC ~€590–600/t EXW, autumn 2025. Source: https://www.spglobal.com/commodity-insights/en/news-research/latest-news/metals/030121-europe-hrc-steel-to-raw-materials-spread-rises-on-steel-prices
    "cement": 135,    # €/t – Europe cement ≈ USD 145 → ~€135/t (approx)
    "ammonia": 555,   # €/t – Europe ammonia ≈ USD 600 → ~€555/t (approx)
    "methanol": 495,  # €/t – Europe methanol contract ≈ USD 535 → ~€495/t (approx)
    "HVC": 1031      # €/t – Naphtha proxy ≈ USD 650/t → ~€590/t. Source: https://businessanalytiq.com/procurementanalytics/index/naphtha-price-index/
}

prices_world = {
    "steel": 370,     # €/t – Global (world benchmark) steel. Source: https://tradingeconomics.com/commodity/steel
    "cement": 40,     # €/t – NE Asia cement ≈ USD 43 → ~€40/t (approx)
    "ammonia": 456,   # €/t – Global ammonia ≈ USD 493/t → ~€456/t (approx)
    "methanol": 315,  # €/t – China methanol ≈ USD 340/t → ~€315/t (approx)
    "HVC": 842        # €/t – Proxy: average of US/China naphtha. Source: https://businessanalytiq.com/procurementanalytics/index/naphtha-price-index/
}

years = [2024, 2030, 2040, 2050]
price_data

# === Plot ===
fig, axes = plt.subplots(1, len(commodities), figsize=(14, 6), sharex=True, sharey=False)

for idx, (commodity, ax) in enumerate(zip(commodities, axes)):
    for scenario in scenarios:
        label = scenario_labels[scenario] if idx == 0 else None
        y_vals = price_data[commodity].loc[scenario].copy()
        y_vals.index = years

        # Replace 2024 value with current European price
        if 2024 in y_vals.index and prices_europe.get(commodity) is not None:
            y_vals.loc[2024] = prices_europe[commodity]

        # Plot main scenario line
        ax.plot(
            years,
            y_vals,
            marker="o",
            linestyle="-",
            label=label,
            color=scenario_colors.get(scenario, 'black')
        )

        # Add dashed "no CO₂ cost" lines for methanol and HVC
        years_short = [2030, 2040, 2050]
        
        if commodity.lower() == "methanol":
            ax.plot(
                years_short,
                price_data["methanol_noco2"].loc[scenario, years_short],
                marker="o",
                linestyle="--",
                color=scenario_colors.get(scenario, 'black'),
                label=None
            )
        elif commodity.upper() == "HVC":
            ax.plot(
                years_short,
                price_data["HVC_noco2"].loc[scenario, years_short],
                marker="o",
                linestyle="--",
                color=scenario_colors.get(scenario, 'black'),
                label=None
            )


    # Add world price as a black dot slightly offset (so it’s visible next to the EU one)
    if prices_world.get(commodity) is not None:
        ax.scatter(
            [2024.3], [prices_world[commodity]],
            color="black",
            s=40,
            marker="o",
            label="World avg price (2024)" if idx == 0 else None
        )
        ax.text(
            2024.7, prices_world[commodity] + (prices_world[commodity] * 0.03),
            f"€{prices_world[commodity]}",
            fontsize=9, color="black"
        )


    # Shaded import range
    if commodity != "cement" and 2050 in years:
        ax.fill_between(
            [2039, 2051],
            [min_import_costs[commodity]] * 2,
            [max_import_costs[commodity]] * 2,
            color="grey",
            alpha=0.3,
            label="Import price range" if idx == 0 else None
        )
        ax.hlines(
            y=min_import_costs[commodity],
            xmin=2039, xmax=2051,
            colors='black', linestyles='-.', linewidth=1
        )
        ax.hlines(
            y=max_import_costs[commodity],
            xmin=2039, xmax=2051,
            colors='black', linestyles='-.', linewidth=1
        )

    # Titles and formatting
    custom_titles = {"hvc": "Plastics (HVC feedstock)"}
    title = custom_titles.get(commodity.lower(), commodity.title())
    ax.set_title(title)
    ax.set_xticks(years)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--')

    if idx == 0:
        ax.set_ylabel("Price [EUR/t]")
        ax.legend(fontsize=8, loc="lower left")

plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("./graphs/commodity_prices_hist.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
import matplotlib.pyplot as plt

# === Commodity list (cement excluded) ===
commodities = ["steel","cement", "ammonia", "methanol", "HVC"]

# === Current price references (€/t) with sources ===
prices_europe = {
    "steel": 600,     # €/t – Europe HRC ~€590–600/t EXW, autumn 2025
    # Source: https://www.spglobal.com/commodity-insights/en/news-research/latest-news/metals/030121-europe-hrc-steel-to-raw-materials-spread-rises-on-steel-prices
    "ammonia": 555,   # €/t – Europe ammonia ≈ USD 600 → ~€555/t (approx)
    "methanol": 495,  # €/t – Europe methanol ≈ USD 535 → ~€495/t (approx)
    "HVC": 1031        # €/t – Naphtha proxy ≈ USD 650/t → ~€590/t
    # Source: https://businessanalytiq.com/procurementanalytics/index/naphtha-price-index/
}

prices_world = {
    "steel": 400,     # €/t – World benchmark steel
    "ammonia": 456,   # €/t – Global ammonia
    "methanol": 315,  # €/t – China methanol
    "HVC": 480        # €/t – Average of US/China naphtha
}

# === Historical world price ranges (2015–2024, €/t) ===
# Source: TradingEconomics, Business Analytiq (converted to €/t)
world_ranges = {
    "steel": (349.48, 718.99),       # https://tradingeconomics.com/commodity/steel
    "ammonia": (318.9, 801.64),      # https://www.spglobal.com/commodity-insights/en/news-research/latest-news/energy-transition/051023-interactive-ammonia-price-chart-natural-gas-feedstock-europe-usgc-black-sea
    "methanol": (50, 400),         # https://www.sciencedirect.com/science/article/pii/S136403211600229X
    "HVC": (500, 1700)              # https://www.sciencedirect.com/science/article/pii/S136403211600229X
}

years = [2024, 2030, 2040, 2050]

# === Plot ===
fig, axes = plt.subplots(1, len(commodities), figsize=(14, 6), sharex=True, sharey=False)

for idx, (commodity, ax) in enumerate(zip(commodities, axes)):
    for scenario in scenarios:
        label = scenario_labels[scenario] if idx == 0 else None
        y_vals = price_data[commodity].loc[scenario].copy()
        y_vals.index = years

        # Replace 2024 value with current European price
        if 2024 in y_vals.index and prices_europe.get(commodity) is not None:
            y_vals.loc[2024] = prices_europe[commodity]

        # Plot main scenario line
        ax.plot(
            years,
            y_vals,
            marker="o",
            linestyle="-",
            label=label,
            color=scenario_colors.get(scenario, 'black')
        )

        # Add dashed "no CO₂ cost" lines for methanol and HVC (only 2030–2050)
        years_short = [2030, 2040, 2050]
        if commodity.lower() == "methanol":
            ax.plot(
                years_short,
                price_data["methanol_noco2"].loc[scenario, years_short],
                marker="o",
                linestyle="--",
                color=scenario_colors.get(scenario, 'black'),
                label=None
            )
        elif commodity.upper() == "HVC":
            ax.plot(
                years_short,
                price_data["HVC_noco2"].loc[scenario, years_short],
                marker="o",
                linestyle="--",
                color=scenario_colors.get(scenario, 'black'),
                label=None
            )

    # Add Europe current price dot + annotation
    """
    if prices_europe.get(commodity) is not None:
        ax.scatter(
            [2024], [prices_europe[commodity]],
            color="blue",
            s=40,
            marker="o",
            label="Europe current price (2024)" if idx == 0 else None
        )
        ax.text(
            2024, prices_europe[commodity] + (prices_europe[commodity] * 0.04),
            f"€{prices_europe[commodity]}",
            fontsize=10, color="blue"
        )
    """
    # Add world price range band (shaded) and annotation
    if commodity in world_ranges:
        low, high = world_ranges[commodity]
        x0, x1 = 2023.8, 2024.2
        # Slightly extend the range visually (±5%)
        low_vis = low * 0.95
        high_vis = high * 1.05

        ax.fill_betweenx(
            y=[low_vis, high_vis],
            x1=x1,
            x2=x0,
            color="red",
            alpha=0.25,
            label="Hist. price range" if idx == 0 else None
        )
        ax.hlines(y=low, xmin=x0, xmax=x1, colors='red', linestyles='--', linewidth=1)
        ax.hlines(y=high, xmin=x0, xmax=x1, colors='red', linestyles='--', linewidth=1)
        ax.text(
            2024.25,
            low_vis - (low_vis * 0.05),   # place the text *below* the lower bound
            f"€{int(low)}–{int(high)}",
            fontsize=10,
            color="red",
            va="top"                      # anchor the text at the top
        )


    # Shaded import range
    if 2050 in years and commodity in min_import_costs:
        ax.fill_between(
            [2039, 2051],
            [min_import_costs[commodity]] * 2,
            [max_import_costs[commodity]] * 2,
            color="grey",
            alpha=0.3,
            label="Import price range" if idx == 0 else None
        )
        ax.hlines(
            y=min_import_costs[commodity],
            xmin=2039, xmax=2051,
            colors='black', linestyles='-.', linewidth=1
        )
        ax.hlines(
            y=max_import_costs[commodity],
            xmin=2039, xmax=2051,
            colors='black', linestyles='-.', linewidth=1
        )

    # Titles and formatting
    custom_titles = {"hvc": "Plastics"}
    title = custom_titles.get(commodity.lower(), commodity.title())
    ax.set_title(title)
    ax.set_xticks(years)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--')

    if idx == 0:
        ax.set_ylabel("Price [EUR/t]")
        ax.legend(fontsize=8, loc="lower left")
    
    
    if commodity.lower() == "methanol":
        # Add line style legend inside the hydrogen subplot
        ax.legend(
            handles=[
                Line2D([0], [0], color='black', linestyle='-', label="Includes EOL CO₂ cost"),
                Line2D([0], [0], color='black', linestyle='--', label="Excludes EOL CO₂ cost")
            ],
            loc="lower right",
            bbox_to_anchor=(1, 0.1),
            fontsize=8,
            frameon=True
        )
        

plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("./graphs/commodity_prices_hist.png", dpi=300, bbox_inches="tight")
plt.show()
