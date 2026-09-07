# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 14:12:13 2025

@author: Dibella
"""

import pypsa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# CONFIG
years = [2030, 2040, 2050]
technologies = ['steel','NH3', 'methanol']#,"H2"]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_april/results/"
lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t

feedstock_colors = {
    
    "Green El H2": "#95C247",
    "Grey El H2": "#96969C",
    "Blue H2": "#2F52E0",
    "Grey H2": "#45454A",
    "Fossil naphtha": "black",
    "Bio naphtha": "#896700",
    "Fischer-Tropsch": "#8A0067",
    #"From methanol": "#FA7E61",
    "With CCS": "#6153CC",
    "Without CCS": "black",
    "Scrap-EAF": "#EEABC4",
    "Green El H2-EAF": "#95C247",  
    "Grey El H2-EAF": "#96969C",
    "Blue H2-EAF": "#2F52E0", 
    "Grey H2-EAF": "#45454A",      
    "CH4-EAF": "#5D461D",         
    "BOF": "black",  
    "BOF + TGR": "#99C0FF",
    "SMR": "black",
    "SMR CC": "#2F52E0",
    "Green El": "#95C247",   # green
    "Grey El": "#96969C",    # brown/grey
    "MeOH import": "#FF8C00",    # Dark Orange
    "NH3 import": "#1E90FF",     # Dodger Blue
    "HBI import-EAF": "#8A2BE2"      # Blue Violet
}


group_countries = {
    'North-West Europe': ['AT', 'BE', 'CH', 'DE', 'FR', 'LU', 'NL','DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE','GB', 'IE'],
    'South Europe': ['ES', 'IT', 'PT', 'GR'],
    'East Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK', 'SI','AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'XK'],
}

country_to_group = {
    country: group for group, countries in group_countries.items() for country in countries
}

def normed(s):
    return s / s.sum()

def get_production(network, tech, target_region):
    timestep = network.snapshot_weightings.iloc[0, 0]
    links = network.links
    p1 = network.links_t.p1

    # Filter links by technology in bus1
    is_tech = links['bus1'].str.contains(tech, case=False, na=False)
    selected_links = links[is_tech].copy()
    if tech == 'methanol':
        selected_links = selected_links.loc[selected_links.index.str.contains('methanolisation'),:]

    if selected_links.empty:
        return 0.0

    # Map link index (country code) to region
    country_codes = selected_links.index.str[:2]
    selected_links['region'] = country_codes.map(country_to_group)

    # Filter for links in the target region
    selected_links = selected_links[selected_links['region'] == target_region]
    if selected_links.empty:
        return 0.0

    # Get production from p1 and sum (MWh)
    total_mwh = -p1[selected_links.index].sum().sum() * timestep 

    # Conversion constants
    lhv_ammonia = 5.166   # MWh / t
    lhv_methanol = 5.528  # MWh / t
    lhv_hydrogen = 33.33 # MWh/ t

    # Convert depending on technology
    if tech.lower() in ['steel']:
        # Divide by 1e3 to convert from GWh to TWh or adjust scale as needed
        # You mentioned "directly" Mt, assuming 1e3 converts to Mt here
        return total_mwh / 1e3  
    elif tech.lower() == 'nh3' or tech.lower() == 'ammonia':
        # Convert MWh to tons using LHV, then to Mt
        return total_mwh / lhv_ammonia / 1e6
    elif tech.lower() == 'methanol':
        return total_mwh / lhv_methanol / 1e6
    elif tech.lower() == 'h2':
        return total_mwh / lhv_hydrogen / 1e6
    else:
        # If tech unknown, just return MWh (or you can return zero)
        return total_mwh
    

def compute_eu_green_share(n):

    # Compute generation from different assets
    timestep = n.snapshot_weightings.iloc[0,0]
    generation_raw = n.generators_t.p.sum() * timestep * 1e-6  # TWh
    generation_raw = generation_raw[~generation_raw.index.str.contains('|'.join(['EU','thermal']), case=False, na=False)]

    hydro_generation = n.storage_units_t.p.sum() * timestep * 1e-6  # TWh
    hydro_generation = hydro_generation[~hydro_generation.index.str.contains('PHS')]

    tot_prod_links = -n.links_t.p1.sum() * timestep * 1e-6
    elec_links_name = ['OCGT', 'coal-', 'CCGT', 'CHP']
    elec_links = tot_prod_links[tot_prod_links.index.str.contains('|'.join(elec_links_name), case=False, na=False)]

    generation = pd.concat([generation_raw, hydro_generation, elec_links])

    # Parse generation index
    generation = (pd.concat([
        generation.index.str.extract(r'([A-Z][A-Z]\d?)', expand=True),          # Node
        generation.index.str.extract(r'([A-Z][A-Z])', expand=True),            # Country
        generation.index.str.extract(r'[A-Z][A-Z]\d? ?\d? (.+)', expand=True),  # Source type
        generation.reset_index()
    ], ignore_index=True, axis=1)
    .drop(3, axis=1)
    .rename(columns={0: 'Node', 1: 'Country', 2: 'Source type', 4: 'Generation [TWh/yr]'}))

    generation = generation[generation['Generation [TWh/yr]'] > 1e-7]

    generation['Source type'] = generation['Source type'].str.split('-').str[0]
    generation['Source type'] = generation['Source type'].str.strip()
    generation['Source type'] = generation['Source type'].str.title()
    generation['Source type'] = generation['Source type'].replace({'Ccgt': 'CCGT', 'Ocgt': 'OCGT'})

    generation = generation.groupby(['Country', 'Source type'])['Generation [TWh/yr]'].sum().reset_index()

    # Compute green share
    green_sources = ['Hydro', 'Ror', 'Solar', 'Wind', 'Biomass',"Nuclear"]
    is_green = generation['Source type'].str.contains('|'.join(green_sources), case=False)

    #gen_by_country = generation.groupby('Country')['Generation [TWh/yr]'].sum()
    #green_gen_by_country = generation[is_green].groupby('Country')['Generation [TWh/yr]'].sum()

    #green_share = (green_gen_by_country / gen_by_country).fillna(0)

    total_eu = generation['Generation [TWh/yr]'].sum()
    green_eu = generation[is_green]['Generation [TWh/yr]'].sum()
    eu_share = green_eu / total_eu if total_eu > 0 else 0
    
    return eu_share


def shares_h2(n):
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Compute green electricity share
    green_share = compute_eu_green_share(n)

    # Define technology categories
    h2_elec  = n.links.loc[n.links.index.str.contains('H2 Electrolysis', regex=True, na=False), :].index
    h2_blue  = n.links.loc[n.links.index.str.contains('SMR CC', regex=True, na=False), :].index
    h2_grey  = n.links.loc[n.links.index.str.contains('SMR(?! CC)', regex=True, na=False), :].index

    # Calculate hydrogen production per category
    h2_elec_df = -n.links_t.p1.loc[:, h2_elec].sum() * timestep
    h2_blue_df = -n.links_t.p1.loc[:, h2_blue].sum() * timestep
    h2_grey_df = -n.links_t.p1.loc[:, h2_grey].sum() * timestep

    # Split electrolysis into green and grey
    h2_elec_green = h2_elec_df * green_share
    h2_elec_grey  = h2_elec_df * (1 - green_share)

    # Aggregate by country code (first two characters of index)
    def aggregate_by_country(df):
        df.index = df.index.str[:2]
        return df.groupby(df.index).sum()

    h2_elec_green = aggregate_by_country(h2_elec_green)
    h2_elec_grey  = aggregate_by_country(h2_elec_grey)
    h2_blue       = aggregate_by_country(h2_blue_df)
    h2_grey       = aggregate_by_country(h2_grey_df)

    # Total hydrogen
    h2_total = h2_elec_green + h2_elec_grey + h2_blue + h2_grey

    # Compute initial shares
    shares = pd.DataFrame({
        "Green El": h2_elec_green / h2_total,
        "Grey El":  h2_elec_grey  / h2_total,
        "Blue":  h2_blue        / h2_total,
        "Grey":  h2_grey        / h2_total
    }).fillna(0)

    # Apply threshold correction (<1% → 0, redistribute)
    threshold = 0.01
    for idx, row in shares.iterrows():
        small = row < threshold
        if small.any():
            keep = ~small
            lost_share = row[small].sum()
            if keep.any() and lost_share > 0:
                shares.loc[idx, keep] += row[keep] / row[keep].sum() * lost_share
            shares.loc[idx, small] = 0

    # Round and add check column
    shares = shares.round(2)
    shares["sum"] = shares.sum(axis=1)

    return shares


def get_steel_prod_eu_import(network):
    timestep = network.snapshot_weightings.iloc[0, 0]

    # Filter steel links excluding heat
    steel_links = network.links[
        (network.links['bus1'].str.contains('steel', case=False, na=False)) &
        (~network.links['bus1'].str.contains('heat', case=False, na=False))
    ].copy()

    if steel_links.empty:
        return pd.Series({
            'Scrap-EAF': 0, 'Green El H2-EAF': 0, "Grey El H2-EAF": 0,
            'Blue H2-EAF': 0, 'Grey H2-EAF': 0, 'CH4-EAF': 0,
            'HBI import-EAF': 0, 'BOF': 0, "BOF + TGR": 0
        })

    # --- Total EAF and BOF ---
    eaf_links = steel_links[steel_links.index.str.contains('EAF')].index
    bof_links = steel_links[steel_links.index.str.contains('BOF')].index

    steel_eaf = -network.links_t.p1.loc[:, eaf_links].sum() * timestep
    steel_bof = -network.links_t.p1.loc[:, bof_links].sum() * timestep

    # --- Emissions capture share for BOF ---
    uncaptured = -network.links_t.p1.filter(like='steel BOF process emis to atmosphere', axis=1).sum().sum() * timestep
    captured = -network.links_t.p2.filter(like='steel BOF CC', axis=1).sum().sum() * timestep
    total_emissions = captured + uncaptured
    share_captured = captured / total_emissions if total_emissions > 0 else 0
    share_uncaptured = 1 - share_captured

    # --- Hydrogen shares ---
    shares = shares_h2(network)

    # --- DRI vs scrap vs HBI ---
    dri_links = -network.links_t.p1.filter(like='DRI-', axis=1).sum() * timestep
    scrap_links = network.links_t.p0.filter(like='steel scrap', axis=1).sum() * timestep
    import_hbi = network.generators_t.p.loc[:, network.generators_t.p.columns.str.contains('HBI import')].sum() * timestep

    total_eaf = steel_eaf.sum()
    total_dri = dri_links.sum()
    total_import = import_hbi.sum()
    total_scrap = scrap_links.sum()
    total = total_dri + total_import + total_scrap

    # Shares (ensure no divide by zero)
    share_scrap = total_scrap / total 
    share_dri = total_dri / total
    share_import = total_import / total

    # --- Within domestic DRI: H2 vs CH4 ---
    dri_ch4 = -network.links_t.p1.filter(like='CH4 to syn gas DRI', axis=1).sum() * timestep
    dri_h2 = -network.links_t.p1.filter(like='H2 to syn gas DRI', axis=1).sum() * timestep
    share_h2 = dri_h2.sum() / (dri_h2.sum() + dri_ch4.sum()) if (dri_h2.sum() + dri_ch4.sum()) > 0 else 0
    share_ch4 = 1 - share_h2

    # --- Production breakdown ---
    scrap_eaf = total_eaf * share_scrap / 1e6
    hbi_import_eaf = total_eaf * share_import / 1e6
    dri_eaf = total_eaf * share_dri / 1e6

    ch4_eaf = dri_eaf * share_ch4

    # H2 splits by origin
    green_h2_eaf = dri_eaf * share_h2 * shares["Green El"].mean() 
    grey_el_h2_eaf = dri_eaf * share_h2 * shares["Grey El"].mean() 
    blue_h2_eaf = dri_eaf * share_h2 * shares["Blue"].mean()
    grey_h2_eaf = dri_eaf * share_h2 * shares["Grey"].mean() 

    # BOF breakdown
    bof_uncapt = steel_bof.sum() * share_uncaptured / 1e6
    bof_capt = steel_bof.sum() * share_captured / 1e6

    return pd.Series({
        'Scrap-EAF': scrap_eaf,
        'Green El H2-EAF': green_h2_eaf,
        'Grey El H2-EAF': grey_el_h2_eaf,
        'Blue H2-EAF': blue_h2_eaf,
        'Grey H2-EAF': grey_h2_eaf,
        'CH4-EAF': ch4_eaf,
        'HBI import-EAF': hbi_import_eaf,
        'BOF': bof_uncapt,
        'BOF + TGR': bof_capt
    })



def get_ammonia_prod_eu_import(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    links = n.links
    p1 = n.links_t.p1

    ammonia_links = links[links['bus1'].str.contains("NH3", case=False, na=False)]
    if ammonia_links.empty:
        return pd.Series({"Elect H2": 0.0, "Blue H2": 0.0,"Grey H2": 0.0})

    total_p1 = -p1[ammonia_links.index].sum(axis=0) * timestep
    total_mwh = total_p1.sum()
    
    import_nh3 = n.generators_t.p.loc[:,n.generators_t.p.columns.str.contains('NH3 import')].sum().sum() * timestep

    shares = shares_h2(n)
    green_ratio = shares["Green El"].mean()
    grey_el_ratio = shares["Grey El"].mean()
    blue_ratio = shares["Blue"].mean()
    grey_ratio = shares["Grey"].mean()

    lhv_ammonia = 5.166  # MWh/t
    total_mt = total_mwh / lhv_ammonia / 1e6
    import_nh3_mt = import_nh3 / lhv_ammonia / 1e6

    return pd.Series({
        "Green El H2": total_mt * green_ratio,
        "Grey El H2": total_mt * grey_el_ratio,
        "Blue H2": total_mt * blue_ratio,
        "Grey H2": total_mt * grey_ratio,
        "NH3 import": import_nh3_mt
    })


def get_methanol_prod_eu_import(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    links = n.links
    p1 = n.links_t.p1

    # Identify production links
    methanol_links = links[links['bus1'].str.contains("methanol", case=False, na=False)]
    import_methanol = methanol_links.loc[methanol_links.index.str.contains('import'),:]
    methanolisation_links = methanol_links.loc[methanol_links.index.str.contains('methanolisation'),:]
    methanol_prod = methanolisation_links.loc[~methanolisation_links.index.str.contains('import'),:]
    
    total_p1 = -p1[methanol_prod.index].sum(axis=0) * timestep
    total_mwh = total_p1.sum()
    
    total_import = -p1[import_methanol.index].sum(axis=0) * timestep
    total_mwh_import = total_import.sum()

    # Estimate shares from hydrogen origin
    shares = shares_h2(n)
    green_ratio = shares["Green El"].mean()
    grey_el_ratio = shares["Grey El"].mean()
    blue_ratio = shares["Blue"].mean()
    grey_ratio = shares["Grey"].mean()

    lhv_methanol = 5.528  # MWh/t
    total_mt = total_mwh / lhv_methanol / 1e6
    import_methanol_mt  = total_mwh_import  / lhv_methanol / 1e6

    return pd.Series({
        "Green El H2": total_mt * green_ratio,
        "Grey El H2": total_mt * grey_el_ratio,
        "Blue H2": total_mt * blue_ratio,
        "Grey H2": total_mt * grey_ratio,
        'MeOH import': import_methanol_mt
    })


def get_subtypes_for_tech(tech):
    tech = tech.lower()
    if tech == "methanol":
        return ["Green El H2", "Grey El H2", "Blue H2", "Grey H2", "MeOH import"]
    elif tech == "nh3" or tech == "ammonia":
        return ["Green El H2", "Grey El H2", "Blue H2", "Grey H2", "NH3 import"]
    elif tech == "steel":
        return ['Scrap-EAF', 'Green El H2-EAF', "Grey El H2-EAF", "Blue H2-EAF", 'Grey H2-EAF', 'CH4-EAF', "HBI import-EAF", 'BOF', 'BOF + TGR']
    else:
        return ["Total"]

def get_tech_production_components(n, tech):
    # Dispatch to the correct internal logic for tech-specific subtype breakdowns
    if tech.lower() == "methanol":
        return get_methanol_prod_eu_import(n)
    elif tech.lower() == "steel":
        return get_steel_prod_eu_import(n)
    elif tech.lower() == "nh3":
        return get_ammonia_prod_eu_import(n)
    

def load_networks(scenarios, years, root_dir, res_dir):
    """
    Loads all networks into a dict keyed by (scenario, year).
    """
    networks = {}
    for scenario in scenarios:
        for year in years:
            path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
            networks[(scenario, year)] = pypsa.Network(path)
    return networks


def plot_total_eu_production_by_tech_reversed(
    networks,
    scenarios,
    scenario_labels,
    technologies,
    years,
    save_path="graphs/european_production_stacked.png"
):
    n_rows = len(technologies)
    n_cols = len(scenarios)
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(3.5 * n_cols, 2.5 * n_rows),
        sharex=True,
        #gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1] + [0.9] * (n_cols - 2)}
    )
    
    if n_cols >= 3:
        ax2 = axes[0, 1]  # Column 2
        ax3 = axes[0, 2]  # Column 3
        x2 = ax2.get_position().x1
        x3 = ax3.get_position().x0
        x_line = (x2 + x3) / 2
    
        fig.lines.append(
            plt.Line2D([x_line, x_line], [0, 1], transform=fig.transFigure, color='black', linewidth=1)
        )
    


    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    bar_width = 7  # full-width bars, no space between years

    for i, tech in enumerate(technologies):  # tech index = row
        for j, scenario in enumerate(scenarios):  # scenario index = col
            ax = axes[i, j]
            tech_data = {subtype: [] for subtype in get_subtypes_for_tech(tech)}

            for year in years:
                net = networks.get((scenario, year))
                if net is None:
                    for subtype in tech_data:
                        tech_data[subtype].append(0.0)
                    continue

                prod_df = get_tech_production_components(net, tech)
                for subtype in tech_data:
                    tech_data[subtype].append(prod_df.get(subtype, 0.0))

            # Stack bars by subtype
            bottom = np.zeros(len(years))
            for subtype in tech_data:
                values = np.array(tech_data[subtype])
                ax.bar(
                    years,
                    values,
                    bottom=bottom,
                    width=bar_width,
                    label=subtype,
                    color=feedstock_colors.get(subtype, 'gray'),
                    edgecolor='black',
                    linewidth=0.3,
                    align='center'
                )
                bottom += values

            if j == 0:
                custom_titles = {
                    "nh3": "Ammonia",
                    "h2": "Hydrogen"
                }
                title = custom_titles.get(tech.lower(), tech.title())
                ax.set_ylabel(f"{title} [Mt/yr]", fontsize=12)
            else:
                ax.set_yticklabels([])  # Remove y-axis tick labels
                ax.tick_params(axis='y', which='both', length=0)  # Optional: remove y-ticks


            # Column titles for each scenario
            if i == 0:
                ax.set_title(scenario_labels[scenario], fontsize=14)

            ax.set_xticks(years)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ymax = tech_ymax.get(tech.lower(), None)
            if ymax:
                ax.set_ylim(0, ymax)

            ax.grid(True, linestyle="--", alpha=0.3)

    # Define subtypes by technology for grouped legend
    legend_tech_groups = {
        "Steel": [
            "Scrap-EAF", "Green El H2-EAF", "Grey El H2-EAF", "Blue H2-EAF", "Grey H2-EAF", "CH4-EAF", "HBI import-EAF", "BOF", "BOF + TGR"
        ],
        "Ammonia": [
            "Green El H2", "Grey El H2", "Blue H2", "Grey H2", "NH3 import"
            ],
        "Methanol":  [
            "Green El H2", "Grey El H2", "Blue H2", "Grey H2", "MeOH import"
            ],
    }
    
    # Build custom legend handles with headers
    legend_handles = []
    for idx, (tech, subtypes) in enumerate(legend_tech_groups.items()):
        # Add fake patch as section title
        legend_handles.append(mpatches.Patch(color='none', label=tech))
        for subtype in subtypes:
            if subtype in feedstock_colors:
                patch = mpatches.Patch(color=feedstock_colors[subtype], label=f"{subtype}")
                legend_handles.append(patch)
        
        # Insert spacer after each group except the last
        if idx < len(legend_tech_groups) - 1:
            legend_handles.append(mpatches.Patch(color='none', label=""))
                

    # Add legend
    fig.legend(
        handles=legend_handles,
        loc='center right',
        bbox_to_anchor=(1.2, 0.5),
        fontsize=11,
        #title="Feedstock by Sector",
        frameon=True,
        handlelength=1.5
    )
    

    #plt.suptitle("Total European Industrial Production by Technology and Feedstock", fontsize=15)
    #plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



# %% 
# 4 SCENARIOS

years = [2030,2040,2050]
scenarios = [
    #"import_base_reg_deindustrial",
    #"import_base_reg_regain",
    "import_policy_reg_deindustrial",
    "import_policy_reg_regain"
]




scenario_labels = {
    #"base_reg_deindustrial": "Current deindustrialization",
    #"import_base_reg_regain": "No climate policy",
    "import_policy_reg_deindustrial": "Continued Decline\nIntermediate Imports",
    "import_policy_reg_regain": "Reindustrialization\nIntermediate Imports"
}


tech_ymax = {
    "steel":210,        
    "nh3": 20,
    "methanol": 50,
    "h2": 160
}

# 1. Load all networks once
networks = load_networks(scenarios, years, root_dir, res_dir)

# %%
plot_total_eu_production_by_tech_reversed(
    networks=networks,
    scenarios=scenarios,
    scenario_labels=scenario_labels,
    technologies=technologies,
    years=years,
    save_path="graphs/european_production_stacked_import.png"
)


# %%

scenarios = [
    "policy_reg_deindustrial",
    "import_policy_reg_deindustrial",
    "policy_reg_regain",
    "import_policy_reg_regain"
]

scenario_labels = {
    "policy_reg_deindustrial": "Continued Decline\nNo Interm. Imports",
    "policy_reg_regain": "Reindustrialization\nNo Interm. Imports",
    "import_policy_reg_deindustrial": "Continued Decline\nIntermediate Imports",
    "import_policy_reg_regain": "Reindustrialization\nIntermediate Imports"
}

scenario_colors = {
    "policy_reg_deindustrial": "#8D2E02",
    "policy_reg_regain": "#156639",
    "import_policy_reg_deindustrial": "#8C47D7",
    "import_policy_reg_regain": "#927F63"
}


years = [2030, 2040, 2050]
data_dict = {
    "Annual system cost [bnEUR/a]": {s: {} for s in scenarios},
    "CO2 Price [EUR/tCO2]": {s: {} for s in scenarios},
}

networks = load_networks(scenarios, years, root_dir, res_dir)

for (scenario, year), n in networks.items():
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Annual system cost (initial objective value in bnEUR/a)
    data_dict["Annual system cost [bnEUR/a]"][scenario][year] = n.objective / 1e9

    # Compute corrected total system cost (capex + opex - wrong costs)
    tsc = n.statistics.capex().sum() / 1e9 + n.statistics.opex().sum() / 1e9
    eaf_2020 = n.links[n.links.index.str.endswith("EAF-2020")]
    wrong_costs = (eaf_2020['p_nom'] * eaf_2020['capital_cost']).sum() / 1e9
    tsc -= wrong_costs

    data_dict["Annual system cost [bnEUR/a]"][scenario][year] = tsc

    # Optional: CO₂ emissions (commented in original code)
    # data_dict["CO2 emissions [MtCO2/yr]"][scenario][year] = (
    #     n.stores.loc['co2 atmosphere', 'e_nom_opt'] / 1e6
    # )

    # CO₂ price
    data_dict["CO2 Price [EUR/tCO2]"][scenario][year] = -n.global_constraints.loc['CO2Limit', 'mu']

# === PLOT ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

# Plot: Annual System Cost
label_cost = "Annual system cost [bnEUR/a]"
for scenario in scenarios:
    ax1.plot(
        years,
        [data_dict[label_cost][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )
ax1.set_ylabel("bnEUR/a")
ax1.set_title("Annual System Cost", fontsize = 14)
ax1.set_xticks(years)
ax1.set_ylim(bottom=0)  # Set y-min to 0
ax1.grid(True, linestyle='--')
ax1.legend(fontsize=12)

# Plot: CO2 Emissions
label_emissions = "CO2 Price [EUR/tCO2]"

for scenario in scenarios:
    ax2.plot(
        years,
        [data_dict[label_emissions][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )
ax2.set_ylabel("EUR/tCO2")
ax2.set_title("Carbon Price", fontsize = 14)
ax2.set_xticks(years)
ax2.set_ylim(bottom=0)  # Ensure y-min is 0
ax2.grid(True, linestyle='--')
#ax2.legend(fontsize=12)

# Final layout
plt.tight_layout()
plt.savefig("./graphs/costs_emissions_import.png", dpi=300)
plt.show()

# %%
# === PLOT ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

# ------------------------
# Plot: Annual System Cost
# ------------------------
label_cost = "Annual system cost [bnEUR/a]"
for scenario in scenarios:
    ax1.plot(
        years,
        [data_dict[label_cost][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )

ax1.set_ylabel("bnEUR/a")
ax1.set_title("Annual System Cost", fontsize=14)
ax1.set_xticks(years)
ax1.set_ylim(bottom=0)
ax1.grid(True, linestyle='--')
ax1.legend(fontsize=11)

def add_difference_line(ax, x, y1, y2, abs_diff, pct_diff):
    """
    Draw vertical line with horizontal ticks and add text annotation
    showing absolute and percentage difference inside a small box.
    """
    # vertical line
    ax.plot([x, x], [y1, y2], color="black", linewidth=1.2)
    # horizontal ticks
    ax.plot([x - 0.2, x + 0.2], [y1, y1], color="black", linewidth=1.2)
    ax.plot([x - 0.2, x + 0.2], [y2, y2], color="black", linewidth=1.2)
    
    # text annotation with box
    ax.text(
        x + 0.4, (y1 + y2) / 2,
        f"{abs_diff:+.1f}\n({pct_diff:+.1f}%)",
        va="center", ha="left", fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )


# --- Differences for each year ---
for x in [y for y in years if y != 2030]:
    # Deindustrialization vs Imports
    y_deind = data_dict[label_cost]["policy_reg_deindustrial"][x]
    y_deind_imp = data_dict[label_cost]["import_policy_reg_deindustrial"][x]
    diff_abs = - (y_deind - y_deind_imp)
    diff_pct = 100 * diff_abs / y_deind   # % relative to no imports
    add_difference_line(ax1, x + 0.15, y_deind, y_deind_imp, diff_abs, diff_pct)

    # Reindustrialization vs Imports
    y_reind = data_dict[label_cost]["policy_reg_regain"][x]
    y_reind_imp = data_dict[label_cost]["import_policy_reg_regain"][x]
    diff_abs = - (y_reind - y_reind_imp)
    diff_pct = 100 * diff_abs / y_reind   # % relative to no imports
    add_difference_line(ax1, x + 0.15, y_reind, y_reind_imp, diff_abs, diff_pct)

# ------------------------
# Plot: Carbon Price
# ------------------------
label_emissions = "CO2 Price [EUR/tCO2]"
for scenario in scenarios:
    ax2.plot(
        years,
        [data_dict[label_emissions][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )

ax2.set_ylabel("EUR/tCO2")
ax2.set_title("Carbon Price", fontsize=14)
ax2.set_xticks(years)
ax2.set_ylim(bottom=0)
ax2.grid(True, linestyle='--')

# Final layout
plt.tight_layout()
plt.savefig("./graphs/costs_emissions_import_withdiff_all_lines.png", dpi=300)
plt.show()

# %%

# === PLOT ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=False)

# ------------------------
# Plot: Annual System Cost (stacked bars)
# ------------------------
label_cost = "Annual system cost [bnEUR/a]"
bar_width = 0.35
x = np.arange(len(years))


# --- Helper to plot No-Import bar with Import difference as dot ---
def plot_no_import_with_import_dot(no_vals, imp_vals, xpos, color_no, color_imp, label_no, label_imp):
    # Plot No Import bars
    ax1.bar(
        xpos, no_vals,
        width=bar_width,
        color=color_no,
        edgecolor="black",
        label=label_no,
        zorder=2,
        alpha = 0.8
    )
    # Plot Import difference as a dot
    for x_pos, no, imp in zip(xpos, no_vals, imp_vals):
        diff = imp - no
        ax1.scatter(
            x_pos, no + diff,          # position at the Import value
            color=color_imp, 
            edgecolor="black",
            s=100,                      # marker size
            zorder=4
        )
        # Optional: annotate difference
        #ax1.text(
        #    x_pos, no + diff + 5,      # slightly above the dot
        #    f"{diff:+.1f}", fontsize=9, ha="center"
        #)
        
        
# --- Values ---
deind_no = [data_dict[label_cost]["policy_reg_deindustrial"][y] for y in years]
deind_imp = [data_dict[label_cost]["import_policy_reg_deindustrial"][y] for y in years]
reind_no = [data_dict[label_cost]["policy_reg_regain"][y] for y in years]
reind_imp = [data_dict[label_cost]["import_policy_reg_regain"][y] for y in years]

# Differences (no imports – imports)
deind_diff = [no - imp for no, imp in zip(deind_no, deind_imp)]
reind_diff = [no - imp for no, imp in zip(reind_no, reind_imp)]

# --- Plot both groups ---
plot_no_import_with_import_dot(
    deind_no, deind_imp, x - bar_width/2,
    scenario_colors["policy_reg_deindustrial"], scenario_colors["import_policy_reg_deindustrial"],
    "Continued Decline\nNo Interm. Imports", "Imports"
)
plot_no_import_with_import_dot(
    reind_no, reind_imp, x + bar_width/2,
    scenario_colors["policy_reg_regain"], scenario_colors["import_policy_reg_regain"],
    "Reindustrialization\nNo Interm. Imports", "Imports"
)

# Labels & formatting
ax1.set_ylabel("bnEUR/a")
ax1.set_title("Annual System Cost", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(years)
ax1.set_ylim(bottom=200)
ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

# Legend
#ax1.legend(fontsize=10, ncol=1, frameon=True)

from matplotlib.lines import Line2D

# --- Get existing bar handles and labels ---
handles, labels = ax1.get_legend_handles_labels()

# Create custom handles for Import dots
dot_deind = Line2D([0], [0], marker='o', color='w', markeredgecolor="black",
                    markerfacecolor=scenario_colors["import_policy_reg_deindustrial"],
                    markersize=8, label="Continued Decline\nIntermediate Imports")
dot_reind = Line2D([0], [0], marker='o', color='w', markeredgecolor="black",
                    markerfacecolor=scenario_colors["import_policy_reg_regain"],
                    markersize=8, label="Reindustrialization\nIntermediate Imports")

# Append to legend
handles.extend([dot_deind, dot_reind])
labels.extend([dot_deind.get_label(), dot_reind.get_label()])

# Update legend
ax1.legend(handles=handles, labels=labels, fontsize=10, ncol=1, frameon=True)


# --- Difference annotations ---
def add_difference_line(ax, xpos, y_no, y_imp, abs_diff, pct_diff):
    ax.text(
        xpos + 0.2, (y_no + y_imp) / 2,
        f"{abs_diff:+.1f} bnEUR\n({pct_diff:+.1f}%)",
        va="center", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )

for i, year in enumerate(years):
    if year == 2030:
        continue
    add_difference_line(ax1, x[i] - bar_width/2,
                        deind_no[i], deind_imp[i],
                        - (deind_no[i] - deind_imp[i]),
                        100 * (-deind_no[i] + deind_imp[i]) / deind_no[i])
    add_difference_line(ax1, x[i] + bar_width/2,
                        reind_no[i], reind_imp[i],
                        -(reind_no[i] - reind_imp[i]),
                        100 * (-reind_no[i] + reind_imp[i]) / reind_no[i])

# ------------------------
# Plot: Carbon Price (lines)
# ------------------------
label_emissions = "CO2 Price [EUR/tCO2]"
for scenario in scenarios:
    ax2.plot(
        years,
        [data_dict[label_emissions][scenario][year] for year in years],
        linestyle="-",
        marker="o",
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )

ax2.set_ylabel("EUR/tCO2")
ax2.set_title("Carbon Price", fontsize=14)
ax2.set_xticks(years)
ax2.set_ylim(bottom=0)
ax2.grid(True, linestyle="--")

# Final layout
plt.tight_layout()
plt.savefig("./graphs/costs_emissions_import_withdiff_all.png", dpi=300)
plt.show()


# %%

from PIL import Image, ImageDraw, ImageFont

# Paths to your two plots
img_path_production = "graphs/european_production_stacked_import.png"
img_path_prices = "graphs_paper/costs_emissions_import_withdiff_all.png"

# Open images
img_production = Image.open(img_path_production)
img_prices = Image.open(img_path_prices)

# Resize images to the same width (optional, adjust as needed)
target_width = max(img_production.width, img_prices.width)
img_production_resized = img_production.resize(
    (target_width, int(img_production.height * target_width / img_production.width)),
    Image.Resampling.LANCZOS
)
img_prices_resized = img_prices.resize(
    (target_width, int(img_prices.height * target_width / img_prices.width)),
    Image.Resampling.LANCZOS
)

# Function to add upper-left labels inside the plot
def add_label(image, label, padding=10):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except:
        font = ImageFont.load_default()
    draw.text((padding, padding), label, fill="black", font=font)
    return image

# Add labels inside the plots at top-left
img_production_labeled = add_label(img_production_resized.copy(), "a)")
img_prices_labeled = add_label(img_prices_resized.copy(), "b)")

# Combine images vertically
combined_height = img_production_labeled.height + img_prices_labeled.height
combined_img = Image.new("RGB", (target_width, combined_height), color=(255, 255, 255))

combined_img.paste(img_production_labeled, (0, 0))
combined_img.paste(img_prices_labeled, (0, img_production_labeled.height))

# Save combined image
combined_img.save("graphs/combined_production_prices_labeled.png")
