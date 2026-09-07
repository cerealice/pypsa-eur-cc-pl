# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:30:54 2025

@author: Dibella
"""

import os
import pypsa
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import yaml
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
import numpy as np

# Mapping of country codes to names
get_country_name = {
    'AT': 'Austria', 'BE': 'Belgium', 'CH': 'Switzerland', 'DE': 'Germany', 'FR': 'France', 'LU': 'Luxembourg',
    'NL': 'Netherlands', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'LV': 'Latvia', 'LT': 'Lithuania',
    'NO': 'Norway', 'SE': 'Sweden', 'GB': 'United Kingdom', 'IE': 'Ireland', 'ES': 'Spain', 'IT': 'Italy',
    'PT': 'Portugal', 'GR': 'Greece', 'BG': 'Bulgaria', 'CZ': 'Czech Republic', 'HU': 'Hungary', 'PL': 'Poland',
    'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia', 'AL': 'Albania', 'BA': 'Bosnia and Herzegovina',
    'HR': 'Croatia', 'ME': 'Montenegro', 'MK': 'North Macedonia', 'RS': 'Serbia', 'XK': 'Kosovo'
}

scenario_labels = {
    "policy_reg_deindustrial": "No relocation",
    "policy_eu_deindustrial": "Relocation",
    "policy_reg_regain": "No relocation",
    "policy_eu_regain": "Relocation"
}

scenario_colors = {
    "base_reg_deindustrial": "#813746",
    "policy_reg_deindustrial": "#FC814A",
    "base_reg_regain": "#6D8088",
    "policy_reg_regain": "#28C76F",
    "import_policy_reg_deindustrial": "#A6A14E",
    "import_policy_reg_regain": "#6BCDC9",
    "policy_eu_deindustrial": "#AA6DA3",
    "policy_eu_regain": "#3C89CD"
}


# === LOADER ===
def load_networks(scenarios, years, root_dir, res_dir):
    """
    Load all networks once into a dict keyed by (scenario, year).
    """
    networks = {}
    for scenario in scenarios:
        for year in years:
            path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
            networks[(scenario, year)] = pypsa.Network(path)
    return networks

def extract_avg_prices_per_country(networks, scenarios, years):

    avg_prices = {}
    for scenario in scenarios:
        avg_prices[scenario] = {}
        for year in years:
            n = networks[(scenario, year)]
            timestep = n.snapshot_weightings.iloc[0,0]

            # Prices at buses (€/MWh), restrict to main nodes
            prices = n.buses_t.marginal_price
            prices = prices.loc[:, prices.columns.str.endswith(" 0")]
            prices.columns = prices.columns.str[:3]

            elec_gen = n.generators[n.generators["bus"].str.endswith(" 0")]
            gen_dispatch = n.generators_t.p.loc[:,elec_gen.index] * timestep
            gen_dispatch.columns = gen_dispatch.columns.str[:3]
            #gen_dispatch = gen_dispatch.T.groupby(gen_dispatch.columns).sum().T    
            
            elec_sto = n.storage_units_t.p.loc[:,n.storage_units_t.p.columns.str.contains('hydro')].sum()
            sto_dispatch = n.storage_units_t.p.loc[:,elec_sto.index] * timestep
            sto_dispatch.columns = sto_dispatch.columns.str[:3]
            #sto_dispatch = sto_dispatch.T.groupby(sto_dispatch.columns).sum().T  
            
            elec_plants = ["CCGT", "OCGT", "CHP", "lignite","nuclear","coal"]
            elec_links = n.links[n.links.index.str.contains("|".join(elec_plants), case= False)]
            lin_dispatch = -n.links_t.p1.loc[:,elec_links.index] * timestep
            lin_dispatch.columns = lin_dispatch.columns.str[:3]
            #lin_dispatch = lin_dispatch.T.groupby(lin_dispatch.columns).sum().T  
            
            tot_dispatch = pd.concat([gen_dispatch, sto_dispatch, lin_dispatch], axis = 1)
            tot_dispatch = tot_dispatch.T.groupby(tot_dispatch.columns).sum().T
            
            tot_cost_country = (tot_dispatch * prices).sum()
            tot_cost_country.index = tot_cost_country.index.str[:2]
            tot_cost_country = tot_cost_country.groupby(level=0).sum()
            tot_dispatch_country = tot_dispatch.sum()
            tot_dispatch_country.index = tot_dispatch_country.index.str[:2]
            tot_dispatch_country = tot_dispatch_country.groupby(level=0).sum()
            
            avg_price_country = tot_cost_country / tot_dispatch_country
            
            avg_prices[scenario][year] = avg_price_country
    
    return avg_prices

def industrial_elec_per_country(networks, scenarios, years):
    
    ind_plants0 = ["EAF","Haber-Bosch","Electrolysis"]
    ind_plants2 = ["methanolisation"]
    ind_plants3 = ["BF-BOF","DRI","BOF CC"]
    ind_plants4 = ["cement TGR","naphtha steam"]

    ind_elec_cons = {}
    for scenario in scenarios:
        ind_elec_cons[scenario] = {}
        for year in years:
            n = networks[(scenario, year)]
            timestep = n.snapshot_weightings.iloc[0,0]

            elec_ind0 = n.links_t.p0.loc[:,n.links_t.p0.columns.str.contains("|".join(ind_plants0))]
            elec_ind2 = n.links_t.p2.loc[:,n.links_t.p2.columns.str.contains("|".join(ind_plants2))]
            elec_ind3 = n.links_t.p3.loc[:,n.links_t.p3.columns.str.contains("|".join(ind_plants3))]
            elec_ind4 = n.links_t.p4.loc[:,n.links_t.p4.columns.str.contains("|".join(ind_plants4))]

            elec_ind = pd.concat([elec_ind0,elec_ind2,elec_ind3,elec_ind4], axis = 1)
            elec_ind.columns = elec_ind.columns.str[:2]
            elec_ind = elec_ind.T.groupby(elec_ind.columns).sum().T
            elec_ind = elec_ind.sum() * timestep

            ind_elec_cons[scenario][year] = elec_ind
    
    return ind_elec_cons

# Function to get capital cost per country
def get_country_costs(n):

    alinks = n.links.loc[n.links['p_nom_extendable'] == True, 'p_nom_opt'] * \
             n.links.loc[n.links['p_nom_extendable'] == True, 'capital_cost']
    
    alinks = alinks[alinks > 0]
    alinks.index = alinks.index.str[:2]
    alinks = alinks.groupby(alinks.index).sum()/1e9
    alinks.index = alinks.index.to_series().map(lambda x: get_country_name.get(x, "Europe"))
    return alinks

def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)

def assign_country(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        c.df["country"] = c.df.index.str[:2]
        
# %%
# === AUTOMATED INDUSTRIAL PRODUCTION MAP ===


# === CONFIGURATION ===
scenarios = [ "policy_reg_deindustrial", "policy_eu_deindustrial", "policy_reg_regain", "policy_eu_regain",]
years = [2030, 2040, 2050]

commodity_tech_dict = {
    "steel": {
        "Green H2 EAF": "EAF",
        "Grey H2 EAF": "EAF",
        "CH4 EAF": "EAF",
        "BOF": "BOF"
    },
    "cement": {
        "Electric kiln": "Electric Kiln",
        "Oxyfuel kiln": "Oxyfuel",
        "Conventional kiln": "Kiln"
    },
    "ammonia": {
        "Green Ammonia": "H2 Electrolysis",
        "SMR Ammonia": "SMR",
        "SMR CC Ammonia": "SMR CC"
    }
}

scenario_labels = {
    "policy_reg_deindustrial": "Continued Decline\nNo relocation",
    "policy_eu_deindustrial": "Continued Decline\nRelocation",
    "policy_reg_regain": "Reindustrialization\nNo relocation",
    "policy_eu_regain": "Reindustrialization\nRelocation"
}



root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_april/results/"
scenario = "base_eu_regain"
regions_fn = root_dir + "resources/" + scenario + "/regions_onshore_base_s_39.geojson"

networks = load_networks(scenarios, years, root_dir, res_dir)

with open(root_dir + res_dir + "base_reg_maintain/configs/config.base_s_39___2030.yaml") as config_file:
    config = yaml.safe_load(config_file)

regions = gpd.read_file(regions_fn).set_index("name")
regions["country"] = regions.index.str[:2]
regions = regions.dissolve(by="country")
config["plotting"]["projection"]["name"] = "EqualEarth"
proj = load_projection(config["plotting"])


# %%
# === TOTAL INDUSTRIAL PRODUCTION MAPS (SUM ALL COMMODITIES) ===
commodities = ["steel", "NH3", "industry methanol", "cement", "hvc"]  #, "H2"]
commodity_search_terms = {
    "steel": "steel",
    "NH3": "NH3",
    "industry methanol": "industry methanol",
    "cement": "cement",
    "hvc": "HVC",
    # "H2": "H2",
}

# Conversion factors from MWh to tons for energy-based commodity buses.
# Steel, cement, and HVC buses are already in t/h so their factor is 1.0.
# NH3 bus is in MWh_NH3 (load set via MWh_NH3_per_tNH3); methanol bus is in MWh_LHV.
commodity_mwh_per_t = {
    "NH3": config["industry"]["MWh_NH3_per_tNH3"],                  # 5.166 MWh/t_NH3
    "industry methanol": config["sector"]["MWh_MeOH_per_tCO2"] * (44 / 32),  # ≈5.54 MWh/t_MeOH (CO2 mol. wt / MeOH mol. wt)
}

max_total_prod = 0
total_prod_results = {}

for (scenario, year), n in networks.items():

    assign_country(n)
    timestep = n.snapshot_weightings.iloc[0, 0]
    total_prod = pd.Series(dtype=float)

    for commodity, search_term in commodity_search_terms.items():
        # Find links matching this commodity
        links = n.links[n.links['bus1'].str.contains(search_term, case=False, na=False)].copy()
        if links.empty:
            continue

        if commodity == "cement":
            # Exclude cement "process emissions" links
            links = links[~links.index.str.contains("process emissions", case=False, na=False)]

        # Assign countries and exclude EU aggregate
        links["country"] = links.index.str[:2]
        # Original (mixed units — NH3 and methanol buses are in MWh, not t):
        # prod = -n.links_t.p1[links.index].sum() * timestep
        # prod.index = prod.index.str[:2]
        # prod = prod.groupby(prod.index).sum() / 1e9  # Gt
        mwh_per_t = commodity_mwh_per_t.get(commodity, 1.0)  # 1.0 for t-based buses (steel, cement, HVC)
        prod = -n.links_t.p1[links.index].sum() * timestep / mwh_per_t
        prod.index = prod.index.str[:2]
        prod = prod.groupby(prod.index).sum() / 1e6  # Mt
        prod = prod[prod.index != "EU"]

        # Accumulate into total production
        total_prod = total_prod.add(prod, fill_value=0)

    # Track global maximum
    max_total_prod = max(max_total_prod, total_prod.max())

    # Save per-scenario/year results
    total_prod_results[(scenario, year)] = total_prod

# %%
max_total_prod = 0
total_prod_results = {}
commodity_prod_results = {}  # (scenario, year) -> DataFrame[country x commodity]
for (scenario, year), n in networks.items():
    assign_country(n)
    timestep = n.snapshot_weightings.iloc[0, 0]
    total_prod = pd.Series(dtype=float)
    commodity_prod = {}  # commodity -> Series[country]
    for commodity, search_term in commodity_search_terms.items():
        # Find links matching this commodity
        links = n.links[n.links['bus1'].str.contains(search_term, case=False, na=False)].copy()
        if links.empty:
            continue
        if commodity == "cement":
            # Exclude cement "process emissions" links
            links = links[~links.index.str.contains("process emissions", case=False, na=False)]
        # Assign countries and exclude EU aggregate
        links["country"] = links.index.str[:2]
        mwh_per_t = commodity_mwh_per_t.get(commodity, 1.0)  # 1.0 for t-based buses (steel, cement, HVC)
        prod = -n.links_t.p1[links.index].sum() * timestep / mwh_per_t
        prod.index = prod.index.str[:2]
        prod = prod.groupby(prod.index).sum() / 1e6  # Mt
        prod = prod[prod.index != "EU"]
        # Store per-commodity series
        commodity_prod[commodity] = prod
        # Accumulate into total production
        total_prod = total_prod.add(prod, fill_value=0)
    # Build country x commodity DataFrame (fills 0 for missing country/commodity combos)
    commodity_prod_results[(scenario, year)] = pd.DataFrame(commodity_prod).fillna(0)
    # Track global maximum
    max_total_prod = max(max_total_prod, total_prod.max())
    # Save per-scenario/year results
    total_prod_results[(scenario, year)] = total_prod

# %%

# EXTRA PART FOR 2024

# --- CONFIGURATION ---
csv_path = '../plots_general/capacities_s_39.csv'


# Load full dataframe (if not already loaded)
df_full = pd.read_csv(csv_path, index_col=0)

# Trim index to first 2 letters (country codes)
df_full.index = df_full.index.str[:2]

# Group by country code (sum all plants in same country)
df_grouped_all = df_full.groupby(df_full.index).sum()

df_by_tec = df_grouped_all.sum()/1e3
# Sum across all technologies and all countries for total 2024 production (kt)
total_prod_kt = df_grouped_all.sum(axis=1)


# Convert to Mt
total_prod_mt = total_prod_kt / 1e3

# For each scenario, sum total production across all countries & technologies for 2024
for scenario in scenarios:
    
    # Store in dictionary with a special key, e.g.:
    total_prod_results[(scenario, 2024)] = total_prod_mt


# %%

#from mpl_toolkits.axes_grid1 import make_axes_locatable

years = [2024,2030,2040,2050]

year_max_total_prod = {}

for year in years:
    max_prod = max(
        total_prod_results[(scenario, year)].max()
        for scenario in scenarios
    )
    year_max_total_prod[year] = max_prod

scenario_max_total_prod = {}

for scenario in scenarios:
    max_prod = max(
        total_prod_results[(scenario, year)].max()
        for year in years
    )
    scenario_max_total_prod[scenario] = max_prod
    


vmin = 0
v1 = 3
v1_plus = 25
v2 = 50
v3 = 70
v4 = 90
vmax = max_total_prod

# Normalize your points between 0 and 1 for colormap creation
points = np.array([vmin, v1, v2, v3, v4, vmax])
normalized_points = (points - vmin) / (vmax - vmin)

# Define your colors at each point (can adjust colors as desired)
colors = ["blue", "lightblue", "white", "pink", "red", "darkred"]

# Create colormap
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    list(zip(normalized_points, colors))
)

# Use Normalize for continuous scaling
norm = Normalize(vmin=vmin, vmax=vmax)
boundaries = [vmin,v1, v1_plus, v2, v3, v4, vmax]
norm = BoundaryNorm(boundaries, ncolors=plt.get_cmap('bwr').N, clip=True)
    


fig2024, ax2024 = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": proj})

# Pick any scenario for 2024 since it’s the same across all
scenario_for_2024 = scenarios[0]
total_prod_2024 = total_prod_results[(scenario_for_2024, 2024)]
regions["total_industrial_prod"] = total_prod_2024
regions_2024 = regions.to_crs(proj.proj4_init)

regions_2024.plot(
    ax=ax2024,
    column="total_industrial_prod",
    cmap="RdYlGn",
    linewidth=0.3,
    edgecolor="black",
    norm=norm,
)

ax2024.set_title("2024", fontsize=16)
ax2024.set_facecolor("white")
ax2024.set_axis_off()

plt.tight_layout()
plt.savefig("graphs/total_industrial_production_2024_single_map.png", dpi=300, bbox_inches='tight')
plt.show()

years_grid = [2030, 2040, 2050]

fig, axes = plt.subplots(
    len(scenarios),
    len(years_grid),
    figsize=(4.5 * len(years_grid), 4.5 * len(scenarios)),
    subplot_kw={"projection": proj}
)

fig.subplots_adjust(right=0.85)

for i, scenario in enumerate(scenarios):
    for j, year in enumerate(years_grid):
        ax = axes[i, j]
        total_prod = total_prod_results[(scenario, year)]
        regions["total_industrial_prod"] = total_prod
        regions_proj = regions.to_crs(proj.proj4_init)

        regions_proj.plot(
            ax=ax,
            column="total_industrial_prod",
            cmap="RdYlGn",
            linewidths=0.3,
            legend=False,
            norm=norm,
            edgecolor="black",
        )
        ax.set_facecolor("white")
        if i == 0:
            ax.set_title(year, fontsize=16, loc='center')

        if j == 0:
            ax.annotate(
                scenario_labels.get(scenario, scenario),
                xy=(-0.1, 0.5), xycoords='axes fraction',
                fontsize=14, ha='center', va='center', rotation=90,
                fontweight='bold'
            )
            
        ax.set_axis_off()

# Add colorbar only once on the right
cax = fig.add_axes([1.02, 0.15, 0.02, 0.7]) 
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Total Industrial Production [Gt/yr]", fontsize=16, labelpad=20)
cbar.ax.tick_params(labelsize=14)

# Add rotated group labels left of y-axis labels (adjust as needed)
left_text_x = -0.02
mid_row_1_2 = 0.75
mid_row_3_4 = 0.25
fig.text(left_text_x, mid_row_1_2, "Continued Decline", rotation=90,
         fontsize=16, fontweight='bold', va='center', ha='center')
fig.text(left_text_x, mid_row_3_4, "Reindustrialization", rotation=90,
         fontsize=16, fontweight='bold', va='center', ha='center')

# Horizontal line between rows 2 and 3
n_rows = len(scenarios)
step = 1 / n_rows
row2_center = 1 - (1 + 0.5) * step
row3_center = 1 - (2 + 0.5) * step
line_y = (row2_center + row3_center) / 2 - (row2_center + row3_center) / 100
fig.lines.append(plt.Line2D([0, 1], [line_y, line_y], transform=fig.transFigure, color='black', linewidth=1.5))

plt.tight_layout()
plt.savefig("graphs/total_industrial_production_per_country_2030_2040_2050.png", dpi=300, bbox_inches='tight')
plt.show()


# %%

# Example inputs
years = [2030, 2040, 2050]  # or whatever years you are using
scenarios = [
    "policy_reg_deindustrial",
    "policy_eu_deindustrial",
    "policy_reg_regain",
    "policy_eu_regain"
]

# Initialize data storage
data = []

for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(cwd))
        file_path = os.path.join(parent_dir, res_dir, scenario, "networks", f"base_s_39___{year}.nc")
        
        n = pypsa.Network(file_path)
        
        annual_cost = n.objective / 1e9  # Convert to billion euros per year
        
        data.append({
            'Year': year,
            'Scenario': scenario,
            'Value': annual_cost
        })

# Convert to DataFrame
df_total_cost = pd.DataFrame(data)

# Map scenario labels (same as in your plotting code)
df_total_cost['Scenario_label'] = df_total_cost['Scenario'].map(scenario_labels)

# Assuming df_total_cost has columns: Year, Scenario, Value
df_total_cost_filtered = df_total_cost[df_total_cost['Scenario'].isin(scenarios)].copy()

# Merge for plotting dots later
df_total_cost_filtered['Scenario_label'] = df_total_cost_filtered['Scenario'].map(scenario_labels)



# %%

years = [2030, 2040, 2050]
data_dict = {
    "Annual system cost [bnEUR/a]": {s: {} for s in scenarios},
    "CO2 Price [EUR/tCO2]": {s: {} for s in scenarios},
}

# === LOAD DATA ===
for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(cwd))
        file_path = os.path.join(parent_dir, res_dir, scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0,0]
        
        data_dict["Annual system cost [bnEUR/a]"][scenario][year] = n.objective / 1e9



# %%
# === PLOT: Relocation with bars + dots ===
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

label_cost = "Annual system cost [bnEUR/a]"
bar_width = 0.35
x = np.arange(len(years))

# --- Helper to plot Relocation bar with No-Relocation as dot ---
def plot_reloc_with_dot(rel_vals, no_vals, xpos, color_reloc, color_no, label_reloc, label_no):
    # Relocation (baseline) bar
    ax1.bar(
        xpos, rel_vals,
        width=bar_width,
        color=color_reloc,
        edgecolor="black",
        label=label_reloc,
        zorder=2,
        alpha=0.8
    )
    # Dot for No Relocation
    for x_pos, rel, no in zip(xpos, rel_vals, no_vals):
        ax1.scatter(
            x_pos, no,                  # dot at the no-relocation value
            color=color_no,
            edgecolor="black",
            s=100,
            zorder=4
        )

# --- Values ---
deind_no = [data_dict[label_cost]["policy_reg_deindustrial"][y] for y in years]
deind_rel = [data_dict[label_cost]["policy_eu_deindustrial"][y] for y in years]
reind_no = [data_dict[label_cost]["policy_reg_regain"][y] for y in years]
reind_rel = [data_dict[label_cost]["policy_eu_regain"][y] for y in years]

# --- Plot both groups ---
plot_reloc_with_dot(
    deind_rel, deind_no, x - bar_width/2,
    scenario_colors["policy_eu_deindustrial"], scenario_colors["policy_reg_deindustrial"],
    "Continued Decline\nRelocation", "No Relocation"
)
plot_reloc_with_dot(
    reind_rel, reind_no, x + bar_width/2,
    scenario_colors["policy_eu_regain"], scenario_colors["policy_reg_regain"],
    "Reindustrialization\nRelocation", "No Relocation"
)

# Labels & formatting
ax1.set_ylabel("bnEUR/a")
ax1.set_title("Annual System Cost", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(years)
ax1.set_ylim(bottom=200)
ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

# --- Custom Legend with dots ---
from matplotlib.lines import Line2D

handles, labels = ax1.get_legend_handles_labels()

# Custom handles for No Relocation dots
dot_deind = Line2D([0], [0], marker='o', color='w', markeredgecolor="black",
                   markerfacecolor=scenario_colors["policy_reg_deindustrial"],
                   markersize=8, label="Continued Decline\nNo Relocation")
dot_reind = Line2D([0], [0], marker='o', color='w', markeredgecolor="black",
                   markerfacecolor=scenario_colors["policy_reg_regain"],
                   markersize=8, label="Reindustrialization\nNo Relocation")

handles.extend([dot_deind, dot_reind])
labels.extend([dot_deind.get_label(), dot_reind.get_label()])

ax1.legend(handles=handles, labels=labels, fontsize=11, ncol=1, frameon=True)

# --- Difference annotations ---
def add_difference_line(ax, xpos, y_no, y_rel, abs_diff, pct_diff):
    ax.text(
        xpos + 0.2, (y_no + y_rel) / 2,
        f"{abs_diff:+.1f} bnEUR\n({pct_diff:+.1f}%)",
        va="center", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )

for i, year in enumerate(years):
    add_difference_line(ax1, x[i] - bar_width/2,
                        deind_no[i], deind_rel[i],
                        deind_no[i] - deind_rel[i],
                        100 * (deind_no[i] - deind_rel[i]) / deind_no[i])
    add_difference_line(ax1, x[i] + bar_width/2,
                        reind_no[i], reind_rel[i],
                        reind_no[i] - reind_rel[i],
                        100 * (reind_no[i] - reind_rel[i]) / reind_no[i])

plt.tight_layout()
plt.savefig("./graphs/costs_reloc_withdiff_dots.png", dpi=300)
plt.show()


# %%

# Extract average electricity prices per country
avg_prices = extract_avg_prices_per_country(networks, scenarios, years)

ind_elec_cons = industrial_elec_per_country(networks, scenarios, years)

# %%


# Compute total spending per scenario and year (already in billions €)
total_spending = {}
for scenario in scenarios:
    total_spending[scenario] = {}
    for year in years:
        prices = avg_prices[scenario][year]
        cons = ind_elec_cons[scenario][year]
        common_countries = prices.index.intersection(cons.index)
        prices = prices.loc[common_countries]
        cons = cons.loc[common_countries]
        spending_country = prices * cons
        total_spending[scenario][year] = spending_country.sum() / 1e9  # B€

# Convert to DataFrame
total_spending_df = pd.DataFrame(total_spending, index=years)
total_spending_df.index.name = "Year"

# --- Plot grouped bar chart ---
fig, ax = plt.subplots(figsize=(8, 8))
width = 0.2
x = np.arange(len(years))

# Map scenario to label for plotting
scenario_label_map = {s: scenario_labels.get(s, s) for s in scenarios}

# Plot bars
for i, scenario in enumerate(scenarios):
    ax.bar(x + i*width, total_spending_df[scenario], width=width, edgecolor='black',linewidth=0.7,
           color=scenario_colors[scenario], label=scenario_label_map[scenario])

# --- Function to add difference boxes ---
def add_difference_box(ax, xpos, val_no, val_reloc):
    diff_abs = val_reloc - val_no
    diff_pct = 100 * diff_abs / val_no
    ax.annotate(
        f"{diff_abs:+.2f} bnEUR\n({diff_pct:+.1f}%)",
        xy=(xpos, val_reloc),
        xytext=(xpos, ((val_no + val_reloc)/2)+10),
        textcoords='data',
        ha='center', va='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
        #arrowprops=dict(arrowstyle="->", color='black', lw=1)
    )

# Add difference boxes between No-relocation and Relocation bars
for j, year in enumerate(years):
    # Deindustrialization pair
    x_pos_deind = x[j] - width/2 + 1.5*width  # center between two bars
    val_no_deind = total_spending_df.loc[year, "policy_reg_deindustrial"]
    val_rel_deind = total_spending_df.loc[year, "policy_eu_deindustrial"]
    add_difference_box(ax, x_pos_deind, val_no_deind, val_rel_deind)
    
    # Reindustrialization pair
    x_pos_reind = x[j] + width/2 + 2.5*width  # center between two bars
    val_no_reind = total_spending_df.loc[year, "policy_reg_regain"]
    val_rel_reind = total_spending_df.loc[year, "policy_eu_regain"]
    add_difference_box(ax, x_pos_reind, val_no_reind, val_rel_reind)

# Final formatting
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(years)
ax.set_ylabel("bnEUR/a")
ax.set_title("Total industrial electricity expenditures", fontsize = 14)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("./graphs/ind_elec_costs_reloc.png", dpi=300)
plt.show()

# %%
scenario_labels = {
    "policy_reg_deindustrial": "Continued Decline\nNo relocation",
    "policy_eu_deindustrial": "Continued Decline\nRelocation",
    "policy_reg_regain": "Reindustrialization\nNo relocation",
    "policy_eu_regain": "Reindustrialization\nRelocation"
}

scenario_colors = {
    "policy_reg_deindustrial": "#FC814A",
    "policy_reg_regain": "#28C76F",
    "policy_eu_deindustrial": "#AA6DA3",
    "policy_eu_regain": "#3C89CD"
}

scenarios = ["policy_reg_deindustrial",
             "policy_eu_deindustrial",
             "policy_reg_regain",
             "policy_eu_regain",]



def extract_avg_prices_per_country(networks, scenarios, years):

    avg_prices = {}
    for scenario in scenarios:
        avg_prices[scenario] = {}
        for year in years:
            n = networks[(scenario, year)]
            timestep = n.snapshot_weightings.iloc[0,0]

            # Prices at buses (€/MWh), restrict to main nodes
            prices = n.buses_t.marginal_price
            prices = prices.loc[:, prices.columns.str.endswith(" 0")]
            prices.columns = prices.columns.str[:3]

            elec_gen = n.generators[n.generators["bus"].str.endswith(" 0")]
            gen_dispatch = n.generators_t.p.loc[:,elec_gen.index] * timestep
            gen_dispatch.columns = gen_dispatch.columns.str[:3]
            #gen_dispatch = gen_dispatch.T.groupby(gen_dispatch.columns).sum().T    
            
            elec_sto = n.storage_units_t.p.loc[:,n.storage_units_t.p.columns.str.contains('hydro')].sum()
            sto_dispatch = n.storage_units_t.p.loc[:,elec_sto.index] * timestep
            sto_dispatch.columns = sto_dispatch.columns.str[:3]
            #sto_dispatch = sto_dispatch.T.groupby(sto_dispatch.columns).sum().T  
            
            elec_plants = ["CCGT", "OCGT", "CHP", "lignite","nuclear","coal"]
            elec_links = n.links[n.links.index.str.contains("|".join(elec_plants), case= False)]
            lin_dispatch = -n.links_t.p1.loc[:,elec_links.index] * timestep
            lin_dispatch.columns = lin_dispatch.columns.str[:3]
            #lin_dispatch = lin_dispatch.T.groupby(lin_dispatch.columns).sum().T  
            
            tot_dispatch = pd.concat([gen_dispatch, sto_dispatch, lin_dispatch], axis = 1)
            tot_dispatch = tot_dispatch.T.groupby(tot_dispatch.columns).sum().T
            
            tot_cost_country = (tot_dispatch * prices).sum()
            tot_cost_country.index = tot_cost_country.index.str[:2]
            tot_cost_country = tot_cost_country.groupby(level=0).sum()
            tot_dispatch_country = tot_dispatch.sum()
            tot_dispatch_country.index = tot_dispatch_country.index.str[:2]
            tot_dispatch_country = tot_dispatch_country.groupby(level=0).sum()
            
            avg_price_country = tot_cost_country / tot_dispatch_country
            
            avg_prices[scenario][year] = avg_price_country
    
    return avg_prices

def industrial_elec_per_country(networks, scenarios, years):
    
    ind_plants0 = ["EAF","Haber-Bosch","Electrolysis"]
    ind_plants2 = ["methanolisation"]
    ind_plants3 = ["BF-BOF","DRI","BOF CC"]
    ind_plants4 = ["cement TGR","naphtha steam"]

    ind_elec_cons = {}
    for scenario in scenarios:
        ind_elec_cons[scenario] = {}
        for year in years:
            n = networks[(scenario, year)]
            timestep = n.snapshot_weightings.iloc[0,0]

            elec_ind0 = n.links_t.p0.loc[:,n.links_t.p0.columns.str.contains("|".join(ind_plants0))]
            elec_ind2 = n.links_t.p2.loc[:,n.links_t.p2.columns.str.contains("|".join(ind_plants2))]
            elec_ind3 = n.links_t.p3.loc[:,n.links_t.p3.columns.str.contains("|".join(ind_plants3))]
            elec_ind4 = n.links_t.p4.loc[:,n.links_t.p4.columns.str.contains("|".join(ind_plants4))]

            elec_ind = pd.concat([elec_ind0,elec_ind2,elec_ind3,elec_ind4], axis = 1)
            elec_ind.columns = elec_ind.columns.str[:2]
            elec_ind = elec_ind.T.groupby(elec_ind.columns).sum().T
            elec_ind = elec_ind.sum() * timestep

            ind_elec_cons[scenario][year] = elec_ind
    
    return ind_elec_cons

"""
years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_april/"
scenario = "base_eu_regain"
regions_fn = root_dir + "resources/" + scenario + "/regions_onshore_base_s_39.geojson"

networks = load_networks(scenarios, years, root_dir, res_dir)
"""
# %%
# Extract average electricity prices per country
avg_prices = extract_avg_prices_per_country(networks, scenarios, years)

ind_elec_cons = industrial_elec_per_country(networks, scenarios, years)

# %%


# Compute total spending per scenario and year (already in billions €)
total_spending = {}
for scenario in scenarios:
    total_spending[scenario] = {}
    for year in years:
        prices = avg_prices[scenario][year]
        cons = ind_elec_cons[scenario][year]
        common_countries = prices.index.intersection(cons.index)
        prices = prices.loc[common_countries]
        cons = cons.loc[common_countries]
        spending_country = prices * cons
        total_spending[scenario][year] = spending_country.sum() / 1e9  # B€

# Convert to DataFrame
total_spending_df = pd.DataFrame(total_spending, index=years)
total_spending_df.index.name = "Year"

# --- Plot grouped bar chart ---
fig, ax = plt.subplots(figsize=(8, 8))
width = 0.2
x = np.arange(len(years))

# Map scenario to label for plotting
scenario_label_map = {s: scenario_labels.get(s, s) for s in scenarios}

# Plot bars
for i, scenario in enumerate(scenarios):
    ax.bar(x + i*width, total_spending_df[scenario], width=width, edgecolor='black',linewidth=0.7,
           color=scenario_colors[scenario], label=scenario_label_map[scenario])

# --- Function to add difference boxes ---
def add_difference_box(ax, xpos, val_no, val_reloc):
    diff_abs = val_reloc - val_no
    diff_pct = 100 * diff_abs / val_no
    ax.annotate(
        f"{diff_abs:+.2f} bnEUR\n({diff_pct:+.1f}%)",
        xy=(xpos, val_reloc),
        xytext=(xpos, ((val_no + val_reloc)/2)+10),
        textcoords='data',
        ha='center', va='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
        #arrowprops=dict(arrowstyle="->", color='black', lw=1)
    )

# Add difference boxes between No-relocation and Relocation bars
for j, year in enumerate(years):
    # Deindustrialization pair
    x_pos_deind = x[j] - width/2 + 1.5*width  # center between two bars
    val_no_deind = total_spending_df.loc[year, "policy_reg_deindustrial"]
    val_rel_deind = total_spending_df.loc[year, "policy_eu_deindustrial"]
    add_difference_box(ax, x_pos_deind, val_no_deind, val_rel_deind)
    
    # Reindustrialization pair
    x_pos_reind = x[j] + width/2 + 2.5*width  # center between two bars
    val_no_reind = total_spending_df.loc[year, "policy_reg_regain"]
    val_rel_reind = total_spending_df.loc[year, "policy_eu_regain"]
    add_difference_box(ax, x_pos_reind, val_no_reind, val_rel_reind)

# Final formatting
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(years)
ax.set_ylabel("bnEUR/a")
ax.set_title("Total industrial electricity expenditures", fontsize = 14)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("./graphs/ind_elec_costs_reloc.png", dpi=300)
plt.show()

            
# %%


# Label function
def add_label(image, label, y_offset=0):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10 + y_offset), label, fill="black", font=font)
    return image

# Paths
img_path_2024 = "graphs/total_industrial_production_2024_single_map.png"
img_path_grid = "graphs/total_industrial_production_per_country_2030_2040_2050.png"
#img_path_costs = "graphs/costs_reloc_withdiff_dots.png"
img_path_costs = "graphs/ind_elec_costs_reloc.png"


# Open images
img_2024 = Image.open(img_path_2024)
img_grid = Image.open(img_path_grid)
img_costs = Image.open(img_path_costs)

# Scale factor for first image
scale_factor = 1.45

# Resize first image
w1_orig, h1_orig = img_2024.size
w1 = int(w1_orig * scale_factor)
h1 = int(h1_orig * scale_factor)
img_2024_resized = img_2024.resize((w1, h1), Image.Resampling.LANCZOS)

# Resize third image to exactly same size as resized first image
img_costs_resized = img_costs.resize((w1, h1), Image.Resampling.LANCZOS)

# Sizes after resizing
w1, h1 = img_2024_resized.size
w3, h3 = img_costs_resized.size

# Target size for image 2 (square)
target_side = h1 + h3

# Resize image 2 to a square
img_grid_square = img_grid.resize((target_side, target_side), Image.Resampling.LANCZOS)

# Add labels to resized versions
img_2024_labeled = add_label(img_2024_resized.copy(), "a)")
img_costs_labeled = add_label(img_costs_resized.copy(), "c)", y_offset=-20)
img_grid_labeled = add_label(img_grid_square.copy(), "b)")

# Recalculate combined canvas size
combined_width = w1 + target_side
combined_height = max(target_side, h1 + h3)

# Create blank white canvas
combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

# Paste images
combined_img.paste(img_2024_labeled, (0, 0))            # top-left (a)
combined_img.paste(img_costs_labeled, (0, h1))          # below a (c)
combined_img.paste(img_grid_labeled, (w1, 0))           # right side (b, now square)

# Save combined image
combined_img.save("graphs/combined_layout_totsyscost.png", dpi=(300, 300))
