# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 08:15:01 2025

@author: Dibella
"""

import pypsa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

# CONFIG
years = [2030, 2040, 2050]
technologies = ['steel', 'cement', 'NH3', 'methanol', 'HVC']#,"H2"]
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
}

group_countries = {
    'North-Western Europe': ['AT', 'BE', 'CH', 'DE', 'FR', 'LU', 'NL','DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE','GB', 'IE'],
    'Southern Europe': ['ES', 'IT', 'PT', 'GR'],
    'Eastern Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK', 'SI','AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'XK'],
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

    if tech == 'H2':
        selected_links = selected_links.loc[~selected_links.index.str.contains('pipeline'),:]

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
    if tech.lower() in ['steel', 'cement', 'hvc']:
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


def share_bio_naphtha(n):
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Get relevant links
    oil_ref = n.generators.index[n.generators.index.str.contains('oil', regex=True, na=False)]
    oil_biomass = n.links.index[n.links.index.str.contains('biomass to liquid', regex=True, na=False)]
    oil_ft = n.links.index[n.links.index.str.contains('Fischer-Tropsch', regex=True, na=False)]

    # Calculate total flows
    oil_bio_total = -n.links_t.p1[oil_biomass].sum().sum() * timestep
    oil_ref_total = n.generators_t.p[oil_ref].sum().sum() * timestep

    # FT flows with country info
    oil_ft_df = -n.links_t.p1[oil_ft].sum(axis=0) * timestep
    oil_ft_df.index = oil_ft_df.index.str[:2]  # extract country code
    oil_ft_by_country = oil_ft_df.groupby(oil_ft_df.index).sum()
    
    # Distribute based on population and GDP
    nuts3 = gpd.read_file('../../resources/base_eu_regain/nuts3_shapes.geojson').set_index("index")
    #nuts3['country'] = nuts3['country'].apply(get_country_name)
    gdp_by_country = nuts3.groupby('country')['gdp'].sum()
    pop_by_country = nuts3.groupby('country')['pop'].sum()
    factors = normed(0.6 * normed(gdp_by_country) + 0.4 * normed(pop_by_country))
    
    oil_bio_by_country = oil_bio_total * factors
    oil_ref_by_country = oil_ref_total * factors
        
    # Compute share
    share_bio = round(oil_bio_by_country / (oil_bio_by_country + oil_ref_by_country + oil_ft_by_country), 2)
    share_ft = round(oil_ft_by_country / (oil_bio_by_country + oil_ref_by_country + oil_ft_by_country), 2)
    return share_bio, share_ft


def get_steel_prod_eu(network):
    timestep = network.snapshot_weightings.iloc[0, 0]

    # Filter steel links excluding heat
    steel_links = network.links[
        (network.links['bus1'].str.contains('steel', case=False, na=False)) &
        (~network.links['bus1'].str.contains('heat', case=False, na=False))
    ].copy()
    steel_links["country"] = steel_links.index.str[:2]

    if steel_links.empty:
        return pd.Series({'Scrap-EAF':0, 'Green El H2-EAF': 0, "Grey El H2-EAF": 0, 'Blue H2-EAF': 0, 'Grey H2-EAF': 0, 'CH4-EAF': 0, 'BOF': 0, "BOF + TGR": 0})

    # Sum steel production per link
    steel_prod = -network.links_t.p1.loc[:, steel_links.index].sum() * timestep
    steel_prod.index = steel_prod.index.str[:2]

    # Get EAF and BOF indices filtered to Europe
    eaf_links = steel_links[steel_links.index.str.contains('EAF')].index
    bof_links = steel_links[steel_links.index.str.contains('BOF')].index

    steel_eaf = -network.links_t.p1.loc[:, eaf_links].sum() * timestep
    steel_bof = -network.links_t.p1.loc[:, bof_links].sum() * timestep

    # Reindex by country code
    steel_eaf.index = steel_eaf.index.str[:2]
    steel_bof.index = steel_bof.index.str[:2]

    # Group sums by country and sum over Europe
    steel_eaf = steel_eaf.groupby(steel_eaf.index).sum()
    steel_bof = steel_bof.groupby(steel_bof.index).sum()
    
    # Emissions not captured and captured by CCS
    uncaptured = -network.links_t.p1.filter(like='steel BOF process emis to atmosphere', axis=1).sum().sum() * timestep
    captured = -network.links_t.p2.filter(like='steel BOF CC', axis=1).sum().sum() * timestep

    total_emissions = captured + uncaptured

    # Share by emissions capture
    share_captured = captured / total_emissions
    share_uncaptured = 1 - share_captured
        
    # Green hydrogen shares by country
    shares = shares_h2(network)

    # DRI vs scrap
    dri_links = -network.links_t.p1.filter(like='DRI-', axis=1).sum() * timestep
    scrap_links = network.links_t.p0.filter(like='steel scrap', axis=1).sum() * timestep
    share_scrap = scrap_links.sum() / (dri_links.sum() + scrap_links.sum()) if (scrap_links.sum() + dri_links.sum()) > 0 else 0
    share_dri = 1 - share_scrap
    
    # In DRI H2 vs CH4
    dri_ch4 = -network.links_t.p1.filter(like='CH4 to syn gas DRI', axis=1).sum() * timestep
    dri_h2 = -network.links_t.p1.filter(like='H2 to syn gas DRI', axis=1).sum() * timestep
    share_h2 = dri_h2.sum() / (dri_h2.sum() + dri_ch4.sum()) if (dri_h2.sum() + dri_ch4.sum()) > 0 else 0
    share_ch4 = dri_ch4.sum() / (dri_h2.sum() + dri_ch4.sum()) if (dri_h2.sum() + dri_ch4.sum()) > 0 else 0

    share_dri_h2 = share_h2 * share_dri
    share_dri_ch4 = share_ch4 * share_dri
    

    # Calculate production categories per country
    steel_prod_scrap_eaf = (steel_eaf * share_scrap).sum() / 1e6  # Mt
    steel_prod_green_el_h2_eaf = (steel_eaf * share_dri_h2 * shares["Green El"]).sum() / 1e6  # Mt
    steel_prod_grey_el_h2_eaf = (steel_eaf * share_dri_h2 * shares["Grey El"]).sum() / 1e6  # Mt
    steel_prod_blue_h2_eaf = (steel_eaf * share_dri_h2 * shares["Blue"]).sum() / 1e6  # Mt
    steel_prod_black_h2_eaf = (steel_eaf * share_dri_h2 * shares["Grey"]).sum() / 1e6
    steel_prod_ch4_eaf = (steel_eaf * share_dri_ch4).sum() / 1e6
    steel_prod_bof_uncapt = steel_bof.sum() * share_uncaptured  / 1e6
    steel_prod_bof_capt = steel_bof.sum() * share_captured  / 1e6

    return pd.Series({
        'Scrap-EAF': steel_prod_scrap_eaf,
        'Green El H2-EAF': steel_prod_green_el_h2_eaf,
        'Grey El H2-EAF': steel_prod_grey_el_h2_eaf,
        'Blue H2-EAF': steel_prod_blue_h2_eaf,
        'Grey H2-EAF': steel_prod_black_h2_eaf,
        'CH4-EAF': steel_prod_ch4_eaf,
        'BOF': steel_prod_bof_uncapt ,
        'BOF + TGR': steel_prod_bof_capt
    })


def get_cement_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Filter cement links (exclude process emissions)
    cement_links = n.links[
        (n.links['bus1'].str.contains('cement', case=False, na=False)) &
        (~n.links['bus1'].str.contains('process emissions', case=False, na=False))
    ]
    if cement_links.empty:
        return pd.Series({'With CCS': 0.0, 'Without CCS': 0.0})

    # Total cement production
    total_cement_mwh = -n.links_t.p1[cement_links.index].sum().sum() * timestep

    # Emissions not captured and captured by CCS
    uncaptured = -n.links_t.p1.filter(like='cement process emis to atmosphere', axis=1).sum().sum() * timestep
    captured = -n.links_t.p2.filter(like='cement TGR', axis=1).sum().sum() * timestep

    total_emissions = captured + uncaptured
    if total_emissions == 0:
        return pd.Series({'With CCS': 0.0, 'Without CCS': 0.0})

    # Share by emissions capture
    share_captured = captured / total_emissions
    share_uncaptured = 1 - share_captured

    # Convert total production from MWh to Mt if needed (optional scaling here)
    cement_total_mt = total_cement_mwh / 1e6

    return pd.Series({
        'With CCS': cement_total_mt * share_captured,
        'Without CCS': cement_total_mt * share_uncaptured
    })



def get_ammonia_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    links = n.links
    p1 = n.links_t.p1

    ammonia_links = links[links['bus1'].str.contains("NH3", case=False, na=False)]
    if ammonia_links.empty:
        return pd.Series({"Elect H2": 0.0, "Blue H2": 0.0,"Grey H2": 0.0})

    total_p1 = -p1[ammonia_links.index].sum(axis=0) * timestep
    total_mwh = total_p1.sum()

    shares = shares_h2(n)
    green_ratio = shares["Green El"].mean()
    grey_el_ratio = shares["Grey El"].mean()
    blue_ratio = shares["Blue"].mean()
    grey_ratio = shares["Grey"].mean()

    lhv_ammonia = 5.166  # MWh/t
    total_mt = total_mwh / lhv_ammonia / 1e6

    return pd.Series({
        "Green El H2": total_mt * green_ratio,
        "Grey El H2": total_mt * grey_el_ratio,
        "Blue H2": total_mt * blue_ratio,
        "Grey H2": total_mt * grey_ratio
    })


def get_methanol_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    links = n.links
    p1 = n.links_t.p1

    # Identify production links
    methanol_links = links[links['bus1'].str.contains("methanol", case=False, na=False)]
    methanol_links = methanol_links.loc[methanol_links.index.str.contains('methanolisation'),:]
    if methanol_links.empty:
        return pd.Series({"Elect H2": 0.0, "Grey H2": 0.0})

    total_p1 = -p1[methanol_links.index].sum(axis=0) * timestep
    total_mwh = total_p1.sum()

    # Estimate shares from hydrogen origin
    shares = shares_h2(n)
    green_ratio = shares["Green El"].mean()
    grey_el_ratio = shares["Grey El"].mean()
    blue_ratio = shares["Blue"].mean()
    grey_ratio = shares["Grey"].mean()

    lhv_methanol = 5.528  # MWh/t
    total_mt = total_mwh / lhv_methanol / 1e6

    return pd.Series({
        "Green El H2": total_mt * green_ratio,
        "Grey El H2": total_mt * grey_el_ratio,
        "Blue H2": total_mt * blue_ratio,
        "Grey H2": total_mt * grey_ratio
    })


def get_hvc_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Select HVC-related links
    hvc_links = n.links[n.links['bus1'].str.contains('HVC', case=False, na=False)]
    if hvc_links.empty:
        return pd.Series({k: 0.0 for k in ["Fossil naphtha", "Bio naphtha", "Fischer-Tropsch", "Methanol"]})

    #hvc_methanol = -n.links_t.p1[hvc_links.index[hvc_links.index.str.contains('methanol')]].sum().sum() * timestep
    hvc_naphtha = -n.links_t.p1[hvc_links.index[hvc_links.index.str.contains('naphtha')]].sum().sum() * timestep

    share_bio, share_ft = share_bio_naphtha(n)
    avg_share_bio = share_bio.mean() if not share_bio.empty else 0.0
    avg_share_ft = share_ft.mean() if not share_ft.empty else 0.0
    share_fossil = max(0.0, 1 - avg_share_bio - avg_share_ft)

    # Convert from kt to Mt (using 1000 divisor)
    #hvc_methanol_mt = hvc_methanol / 1e6
    hvc_naphtha_mt = hvc_naphtha / 1e6

    return pd.Series({
        "Fossil naphtha": hvc_naphtha_mt * share_fossil,
        "Bio naphtha": hvc_naphtha_mt * avg_share_bio,
        "Fischer-Tropsch": hvc_naphtha_mt * avg_share_ft,
        #"From methanol": hvc_methanol_mt
    })


def get_h2_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    lhv_hydrogen = 33.33  # MWh/tH2 (lower heating value)

    tech_names = ['Electrolysis', 'SMR CC', 'SMR']
    tech_productions = {}

    # Compute EU green electricity share
    green_share = compute_eu_green_share(n)

    for tech in tech_names:
        # Match links by technology
        tech_links = n.links[
            n.links.index.str.contains(tech, case=False, na=False)
        ].index

        if tech_links.empty:
            if tech == "Electrolysis":
                tech_productions["Green El"] = 0.0
                tech_productions["Grey El"] = 0.0
            else:
                tech_productions[tech] = 0.0
            continue

        prod = -n.links_t.p1[tech_links].sum().sum() * timestep  # in MWh
        fuel_cells = n.links_t.p0.loc[:, n.links_t.p0.columns.str.contains('Fuel Cell')].sum().sum() * timestep

        net_prod_mt = (prod - fuel_cells) / lhv_hydrogen / 1e6  # Mt H2

        if tech == "Electrolysis":
            tech_productions["Green El"] = net_prod_mt * green_share
            tech_productions["Grey El"] = net_prod_mt * (1 - green_share)
        else:
            tech_productions[tech] = net_prod_mt

    return pd.Series(tech_productions)


def get_subtypes_for_tech(tech):
    tech = tech.lower()
    if tech == "methanol":
        return ["Green El H2", "Grey El H2", "Blue H2", "Grey H2"]
    elif tech == "hvc":
        return ["Fossil naphtha", "Bio naphtha", "Fischer-Tropsch"]#, "From methanol"]
    elif tech == "nh3" or tech == "ammonia":
        return ["Green El H2", "Grey El H2", "Blue H2", "Grey H2"]
    elif tech == "steel":
        return ['Scrap-EAF', 'Green El H2-EAF', "Grey El H2-EAF", "Blue H2-EAF", 'Grey H2-EAF', 'CH4-EAF', 'BOF', 'BOF + TGR']
    elif tech == "cement":
        return ["Without CCS", "With CCS"]
    elif tech == "h2":
        return ["SMR", "SMR CC", "Green El", "Green El"]
    else:
        return ["Total"]

def get_tech_production_components(n, tech):
    # Dispatch to the correct internal logic for tech-specific subtype breakdowns
    if tech.lower() == "methanol":
        return get_methanol_prod_eu(n)
    elif tech.lower() == "steel":
        return get_steel_prod_eu(n)
    elif tech.lower() == 'cement':
        return get_cement_prod_eu(n)
    elif tech.lower() == "hvc":
        return get_hvc_prod_eu(n)
    elif tech.lower() == "nh3":
        return get_ammonia_prod_eu(n)
    elif tech.lower() == "h2":
        return get_h2_prod_eu(n)

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
    nice_scenario_names,
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
    
        
    # ---- Highlight column groups ----
    # Get vertical extent (y0 = bottom of last row, y1 = top of first row)
    y0 = axes[-1, 0].get_position().y0
    y1 = axes[0, 0].get_position().y1
    
    # First column extent
    x0_col1 = axes[0, 0].get_position().x0
    x1_col1 = axes[0, 0].get_position().x1
    
    # Other columns extent (2–end)
    x0_cols = axes[0, 1].get_position().x0
    x1_cols = axes[0, -1].get_position().x1
    
    # Add light gray background behind column 1
    fig.add_artist(Rectangle(
        (x0_col1, y0),  # bottom left corner
        x1_col1 - x0_col1,  # width
        y1 - y0,  # height
        transform=fig.transFigure,
        color="black",
        alpha=0.2,
        zorder=0
    ))
    
    # Add light green background behind the other columns
    fig.add_artist(Rectangle(
        (x0_cols, y0),
        x1_cols - x0_cols,
        y1 - y0,
        transform=fig.transFigure,
        color="green",
        alpha=0.2,
        zorder=0
    ))
    
    if n_cols >= 2:
        ax2 = axes[0, 0]  # Column 1
        ax3 = axes[0, 1]  # Column 2
    
        x2 = ax2.get_position().x1
        x3 = ax3.get_position().x0
        x_line = (x2 + x3) / 2
    
        # Define vertical extent (cut) using top and bottom of first row
        y_top = axes[0, 0].get_position().y1 * 1.05
        y_bottom = axes[-1, 0].get_position().y0 * 0.95
        
        fig.lines.append(
            plt.Line2D([x_line, x_line], [y_bottom, y_top], transform=fig.transFigure, color='black', linewidth=1)
        )
        

    
    # Add group titles above columns
    if n_cols >= 2:
        # Center above column 1
        x_first_group = (axes[0, 0].get_position().x0 + axes[0, 0].get_position().x1) / 2
        # Span columns 2–n (climate policy group)
        x_second_group = (axes[0, 1].get_position().x0 + axes[0, -1].get_position().x1) / 2
    
        y_group_title = axes[0, 0].get_position().y1 + 0.04  # a bit above top row
    
        fig.text(x_first_group, y_group_title, "No Climate Policy", 
                 ha='center', va='bottom', fontsize=14, fontweight="bold")
        fig.text(x_second_group, y_group_title, "Climate Policy", 
                 ha='center', va='bottom', fontsize=14, fontweight="bold")



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
                    "hvc": "Plastics",
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
                ax.set_title(nice_scenario_names[scenario], fontsize=14)

            ax.set_xticks(years)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ymax = tech_ymax.get(tech.lower(), None)
            if ymax:
                ax.set_ylim(0, ymax)

            ax.grid(True, linestyle="--", alpha=0.3)

    # Define subtypes by technology for grouped legend
    legend_tech_groups = {
        "Steel": [
            "Scrap-EAF", "Green El H2-EAF", "Grey El H2-EAF", "Blue H2-EAF", "Grey H2-EAF", "CH4-EAF", "BOF", "BOF + TGR"
        ],
        "Cement": [
            "With CCS","Without CCS"
        ],
        "Ammonia": [
            "Green El H2", "Grey El H2", "Blue H2", "Grey H2"
            ],
        "Methanol":  [
            "Green El H2", "Grey El H2", "Blue H2", "Grey H2"
            ],
        
        "HVC": [
            "Bio naphtha","Fischer-Tropsch","Fossil naphtha" # "From methanol"
            ],
        #"Hydrogen": [
        #    "Green El", "Grey El","SMR CC", "SMR",
        #    ],

    }
    
    # Build custom legend handles with headers and spacing
    legend_handles = []
    for i, (tech, subtypes) in enumerate(legend_tech_groups.items()):
        # Add fake patch as section title
        legend_handles.append(mpatches.Patch(color='none', label=tech))
        for subtype in subtypes:
            if subtype in feedstock_colors:
                patch = mpatches.Patch(color=feedstock_colors[subtype], label=f"{subtype}")
                legend_handles.append(patch)
        
        # Insert spacer after each group except the last
        if i < len(legend_tech_groups) - 1:
            legend_handles.append(mpatches.Patch(color='none', label=""))  # blank line

    
    # Add legend
    fig.legend(
        handles=legend_handles,
        loc='center right',
        bbox_to_anchor=(1.03, 0.5),
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

scenarios = [
    "base_reg_maintain",
    "policy_reg_deindustrial",
    "policy_reg_maintain",
    "policy_reg_regain"
]

nice_scenario_names = {
    "policy_reg_deindustrial": "Continued Decline",
    "policy_eu_regain": "Reindustrial EU",
    "base_reg_maintain": "Stabilization",
    "base_reg_regain": "Base Reindustr",
    "policy_reg_regain": "Reindustrialization",
    "policy_reg_maintain": "Stabilization"
}



tech_ymax = {
    "steel": 210,        
    "cement": 260,
    "nh3": 20,
    "methanol": 50,
    "hvc": 80,
    #"h2": 150
}

# 1. Load all networks once
networks = load_networks(scenarios, years, root_dir, res_dir)

# %% 
plot_total_eu_production_by_tech_reversed(
    networks=networks,
    scenarios=scenarios,
    nice_scenario_names=nice_scenario_names,
    technologies=technologies,
    years=years,
    save_path="graphs/european_production_stacked_wmaintain.png"
)
