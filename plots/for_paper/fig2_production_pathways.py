# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:51:44 2025

@author: Dibella
"""
import pandas as pd
import matplotlib.pyplot as plt

# File paths
scenarios_file = "eu_industry_prod_scenarios.csv"
hist_file = "histo_indu_prod_4graph.xlsx"

# Read scenario projections
df = pd.read_csv(scenarios_file)

# Read Excel and make 'sector' the index immediately
hist_df = pd.read_excel(hist_file, index_col="sector")

# Stack the wide-year columns into a long format (result is a Series with MultiIndex (sector, year))
hist_long = hist_df.stack().rename("value").to_frame()

# Ensure the index names are meaningful
hist_long.index.names = ["sector", "year"]

# Reset only the 'year' level so 'sector' remains the index and 'year' becomes a column
hist_long = hist_long.reset_index()

# Convert year to numeric (if needed)
hist_long["year"] = pd.to_numeric(hist_long["year"], errors="coerce")

# Get unique sector values from scenarios
sectors = df['sector'].unique()

# Exclude chlorine if not needed
sectors_to_plot = [sector for sector in sectors if sector != 'chlorine']

# Create subplots
fig, axes = plt.subplots(1, len(sectors_to_plot), figsize=(len(sectors_to_plot) * 4, 6), sharex=False)

# Ensure axes is iterable
if len(sectors_to_plot) == 1:
    axes = [axes]

# Iterate over each unique sector to create the corresponding subplot
for i, sector in enumerate(sectors_to_plot):
    sector_df = df[df['sector'] == sector]

    # Plot scenario projections
    axes[i].plot(sector_df['year'], sector_df['regain'], 
                 label='Reindustrialization', color='#28C76F')


    # Plot all available historical points
    sector_hist = hist_long[hist_long['sector'] == sector].sort_values("year")
    if not sector_hist.empty:
        # --- Add constant "Current Level" line ---
        last_hist_val = sector_hist['value'].iloc[-1]
        axes[i].hlines(last_hist_val, xmin=sector_df['year'].min(), xmax=sector_df['year'].max(),
                       colors='#DECA4B', linestyles='-', linewidth=2, label='Stabilization')
        axes[i].plot(sector_df['year'], sector_df['deindustrial'], 
                     label="Continued Decline", color='#FC814A')
        axes[i].scatter(sector_hist['year'], sector_hist['value'], 
                        color='black', marker='o', s=40, label='Historical Data')
        axes[i].plot(sector_hist['year'], sector_hist['value'], 
                     color='black', linestyle='--', linewidth=1)




    # Set title
    title = 'Plastics' if sector == 'hvc' else sector.capitalize()
    axes[i].set_title(title, fontsize=14)

    # Y-axis labels
    if sector == 'steel':
        axes[i].set_ylabel('Mt steel/yr', fontsize=12)
    elif sector == 'cement':
        axes[i].set_ylabel('Mt cement/yr', fontsize=12)
    elif sector == 'ammonia':
        axes[i].set_ylabel('Mt NH3/yr', fontsize=12)
    elif sector == 'hvc':
        axes[i].set_ylabel('Mt HVC/yr', fontsize=12)
    elif sector == 'methanol':
        axes[i].set_ylabel('Mt methanol/yr', fontsize=12)

    axes[i].set_ylim(bottom=0)
    axes[i].grid(True, linestyle='--')

    # Only add legend to first subplot
    # --- After plotting all lines ---
    handles, labels = axes[i].get_legend_handles_labels()
    
    # Reorder legend: Reindustrialization, Current Level, Deindustrialization, Historical Data
    legend_order = ['Reindustrialization', 'Stabilization', 'Continued Decline', 'Historical Data']
    ordered_handles = [handles[labels.index(l)] for l in legend_order if l in labels]
    
    

    if i == 0:
        axes[i].legend(ordered_handles, [l for l in legend_order if l in labels])


    
    # Set title
    title = 'Plastics' if sector == 'hvc' else sector.capitalize()
    axes[i].set_title(title, fontsize=14)
    
    # Y-axis labels
    if sector == 'steel':
        axes[i].set_ylabel('Mt steel/yr', fontsize=12)
    elif sector == 'cement':
        axes[i].set_ylabel('Mt cement/yr', fontsize=12)
    elif sector == 'ammonia':
        axes[i].set_ylabel('Mt NH3/yr', fontsize=12)
    elif sector == 'hvc':
        axes[i].set_ylabel('Mt HVC/yr', fontsize=12)
    elif sector == 'methanol':
        axes[i].set_ylabel('Mt methanol/yr', fontsize=12)
        
    axes[i].set_ylim(bottom=0)
    axes[i].grid(True, linestyle='--')
    
    if i == 0:
        axes[i].legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig("graphs/production_projections_with_full_historical.png", bbox_inches="tight")
plt.show()
