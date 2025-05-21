import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pickle
from atb_data import values_from_ATB2024
import os 
from matplotlib.lines import Line2D

ESBWR_Power_Rating = 1500 #MW
BWRX_Power_Rating = 300 #MW
num_simulations = 10000
num_projects_ESBWR = 100
num_projects_BWRX = ESBWR_Power_Rating / BWRX_Power_Rating * num_projects_ESBWR #number of projects for BWRX to reach the same power rating as ESBWR
#num_projects_BWRX = num_projects_ESBWR # equal number of projects for both reactors

concrete_nuclear_cost = 640 #$/m^3
concrete_non_nuclear_cost = 428 #$/m^3
concrete_market_cost = 400 #$/m^3
steel_nuclear_cost = 6743.87 #$/MT
steel_non_nuclear_cost = 3045.33 #$/MT
steel_market_cost = 1000 #$/MT

ESBWR_specific_volume_concrete = 69 #m^3/MW
ESBWR_specific_volume_steel = 33 #MT/MW
BWRX_construction_time_NOAK = [24,36] #months
BWRX_specific_volume_concrete = 0.5 * ESBWR_specific_volume_concrete #50% less concrete per MW 
BWRX_specific_volume_steel = 0.5 * ESBWR_specific_volume_steel #similar decrease as in concrete similar but the ratio has to determined

#Theoretical Material Input Cost (mic) $/kW
ESBWR_mic_nuclear = (ESBWR_specific_volume_concrete * concrete_nuclear_cost + ESBWR_specific_volume_steel * steel_nuclear_cost)/1000 #$/kW
ESBWR_mic_non_nuclear = (ESBWR_specific_volume_concrete * concrete_non_nuclear_cost + ESBWR_specific_volume_steel * steel_non_nuclear_cost) /1000 #$/kW
ESBWR_mic_market = (ESBWR_specific_volume_concrete * concrete_market_cost + ESBWR_specific_volume_steel * steel_market_cost) /1000 #$/kW
BWRX_mic_nuclear = (BWRX_specific_volume_concrete * concrete_nuclear_cost + BWRX_specific_volume_steel * steel_nuclear_cost) /1000 #$/kW
BWRX_mic_non_nuclear = (BWRX_specific_volume_concrete * concrete_non_nuclear_cost + BWRX_specific_volume_steel * steel_non_nuclear_cost) /1000 #$/kW
BWRX_mic_market = (BWRX_specific_volume_concrete * concrete_market_cost + BWRX_specific_volume_steel * steel_market_cost) /1000 #$/kW

#LABOR
#Installation rates
Inst_rate_substructure_concrete_nuclear = 11.45 #hr/m^3
Inst_rate_substructure_concrete_non_nuclear = 5.59 #hr/m^3
Inst_rate_superstructure_concrete_nuclear = 18.51 #hr/m^3
Inst_rate_superstructure_concrete_non_nuclear = 13.88 #hr/m^3
Inst_rate_structural_steel_nuclear = 58.06 #hr/MT
Inst_rate_structural_steel_non_nuclear = 13.06 #hr/MT
# Reactor-specific substructure/superstructure ratios
ESBWR_substructure_percentage = 0.63
ESBWR_superstructure_percentage = 0.37
BWRX_substructure_percentage = 0.47
BWRX_superstructure_percentage = 0.53
#composite wage (Adjusted for inflation 2007-> 2024)
com_wage_concrete = 58.53 #$/hr
com_wage_steel = 66.76 #$/hr

#Construction time for large and SMRs from INL meta-analysis
construction_time_nuclear = [60,82,125] #months
construction_time_smr = [43,55,71] #months


#installation time versus construction time for large and SMRs
inst_time_nuclear_premium = Inst_rate_structural_steel_nuclear * ESBWR_specific_volume_steel * ESBWR_Power_Rating +Inst_rate_substructure_concrete_nuclear * ESBWR_specific_volume_concrete * ESBWR_Power_Rating * ESBWR_substructure_percentage + Inst_rate_superstructure_concrete_nuclear * ESBWR_specific_volume_concrete * ESBWR_Power_Rating * ESBWR_superstructure_percentage #hr
inst_time_nuclear_non_premium = Inst_rate_structural_steel_non_nuclear * ESBWR_specific_volume_steel * ESBWR_Power_Rating +Inst_rate_substructure_concrete_non_nuclear * ESBWR_specific_volume_concrete * ESBWR_Power_Rating * ESBWR_substructure_percentage + Inst_rate_superstructure_concrete_non_nuclear * ESBWR_specific_volume_concrete * ESBWR_Power_Rating * ESBWR_superstructure_percentage #hr
inst_time_smr_premium = Inst_rate_structural_steel_nuclear * BWRX_specific_volume_steel * BWRX_Power_Rating +Inst_rate_substructure_concrete_nuclear * BWRX_specific_volume_concrete * BWRX_Power_Rating * BWRX_substructure_percentage + Inst_rate_superstructure_concrete_nuclear * BWRX_specific_volume_concrete * BWRX_Power_Rating * BWRX_superstructure_percentage #hr
inst_time_smr_non_premium = Inst_rate_structural_steel_non_nuclear * BWRX_specific_volume_steel * BWRX_Power_Rating +Inst_rate_substructure_concrete_non_nuclear * BWRX_specific_volume_concrete * BWRX_Power_Rating * BWRX_substructure_percentage + Inst_rate_superstructure_concrete_non_nuclear * BWRX_specific_volume_concrete * BWRX_Power_Rating * BWRX_superstructure_percentage #hr

average_workload_nuc = 2500 * 2000 #worker x hr/yr
average_workload_smr = 1000 * 2000 #worker x hr/yr
#Average construction time for large and SMRs
inst_time_nuclear_premium_continuous_months = inst_time_nuclear_premium / average_workload_nuc * 12 #months
inst_time_nuclear_non_premium_continuous_months = inst_time_nuclear_non_premium / average_workload_nuc * 12 #months
inst_time_smr_premium_continuous_months = inst_time_smr_premium / average_workload_smr * 12 #months
inst_time_smr_non_premium_continuous_months = inst_time_smr_non_premium / average_workload_smr * 12 #months

inst_time_nuclear_premium_tot_months = inst_time_nuclear_premium_continuous_months * 3 #months
inst_time_nuclear_non_premium_tot_months = inst_time_nuclear_non_premium_continuous_months * 3 #months
inst_time_smr_premium_tot_months = inst_time_smr_premium_continuous_months * 3 #months
inst_time_smr_non_premium_tot_months = inst_time_smr_non_premium_continuous_months * 3 #months

print("ESBWR nuclear premium:", "manhours", inst_time_nuclear_premium, "months", inst_time_nuclear_premium_continuous_months, "total months", inst_time_nuclear_premium_tot_months)
print("ESBWR non-nuclear premium:", "manhours", inst_time_nuclear_non_premium, "months", inst_time_nuclear_non_premium_continuous_months, "total months", inst_time_nuclear_non_premium_tot_months)
print("BWRX nuclear premium:", "manhours", inst_time_smr_premium, "months", inst_time_smr_premium_continuous_months, "total months", inst_time_smr_premium_tot_months)
print("BWRX non-nuclear premium:", "manhours", inst_time_smr_non_premium, "months", inst_time_smr_non_premium_continuous_months, "total months", inst_time_smr_non_premium_tot_months)

#Labor cost for installation of concrete and steel $/kW
# ESBWR labor cost per kW
ESBWR_inst_cost_nuclear = (
    (Inst_rate_substructure_concrete_nuclear * ESBWR_substructure_percentage +
     Inst_rate_superstructure_concrete_nuclear * ESBWR_superstructure_percentage) *
    ESBWR_specific_volume_concrete * com_wage_concrete +
    Inst_rate_structural_steel_nuclear * ESBWR_specific_volume_steel * com_wage_steel
) / 1000

ESBWR_inst_cost_non_nuclear = (
    (Inst_rate_substructure_concrete_non_nuclear * ESBWR_substructure_percentage +
     Inst_rate_superstructure_concrete_non_nuclear * ESBWR_superstructure_percentage) *
    ESBWR_specific_volume_concrete * com_wage_concrete +
    Inst_rate_structural_steel_non_nuclear * ESBWR_specific_volume_steel * com_wage_steel
) / 1000

# BWRX labor cost per kW
BWRX_inst_cost_nuclear = (
    (Inst_rate_substructure_concrete_nuclear * BWRX_substructure_percentage +
     Inst_rate_superstructure_concrete_nuclear * BWRX_superstructure_percentage) *
    BWRX_specific_volume_concrete * com_wage_concrete +
    Inst_rate_structural_steel_nuclear * BWRX_specific_volume_steel * com_wage_steel
) / 1000

BWRX_inst_cost_non_nuclear = (
    (Inst_rate_substructure_concrete_non_nuclear * BWRX_substructure_percentage +
     Inst_rate_superstructure_concrete_non_nuclear * BWRX_superstructure_percentage) *
    BWRX_specific_volume_concrete * com_wage_concrete +
    Inst_rate_structural_steel_non_nuclear * BWRX_specific_volume_steel * com_wage_steel
) / 1000


print("ESBWR installation cost nuclear:", ESBWR_inst_cost_nuclear, "non-nuclear:", ESBWR_inst_cost_non_nuclear)
print("BWRX installation cost nuclear:", BWRX_inst_cost_nuclear, "non-nuclear:", BWRX_inst_cost_non_nuclear)

#Material and Labor cost
ESBWR_total_cost_nuclear = ESBWR_mic_nuclear + ESBWR_inst_cost_nuclear #$/kW
ESBWR_total_cost_non_nuclear = ESBWR_mic_non_nuclear + ESBWR_inst_cost_non_nuclear #$/kW
ESBWR_total_cost_market = ESBWR_mic_market + ESBWR_inst_cost_non_nuclear #$/kW
BWRX_total_cost_nuclear = BWRX_mic_nuclear + BWRX_inst_cost_nuclear #$/kW
BWRX_total_cost_non_nuclear = BWRX_mic_non_nuclear + BWRX_inst_cost_non_nuclear #$/kW
BWRX_total_cost_market = BWRX_mic_market + BWRX_inst_cost_non_nuclear #$/kW


comparison_data=pd.DataFrame({
    "Reactor": ["ESBWR", "BWRX"],
    "Capacity": [ESBWR_Power_Rating, BWRX_Power_Rating],
    "MIC_Market": [ESBWR_mic_market, BWRX_mic_market],
    "MIC_No_Premium": [ESBWR_mic_non_nuclear, BWRX_mic_non_nuclear],
    "MIC_Premium": [ESBWR_mic_nuclear, BWRX_mic_nuclear],
    "Inst_cost_premium": [ESBWR_inst_cost_nuclear, BWRX_inst_cost_nuclear],
    "Inst_cost_no_premium": [ESBWR_inst_cost_non_nuclear, BWRX_inst_cost_non_nuclear],
})

with open("comparison_data.pkl", "wb") as f:
    pickle.dump(comparison_data, f)

# Define colors
colors = {
    "Premium": "#4878A8",
    "No Premium": "#59935B",
    "Market": "#FFA500",
}

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(comparison_data))

for i in x:
    row = comparison_data.iloc[i]
    mic_vals = [row["MIC_Market"], row["MIC_No_Premium"], row["MIC_Premium"]]
    
    # Shaded range
    ax.fill_between([i - 0.2, i + 0.2], min(mic_vals), max(mic_vals), color='lightgray', alpha=0.4)

    # Data points
    ax.scatter(i, row["MIC_Premium"], color=colors["Premium"], s=50, zorder=3)
    ax.text(i + 0.1, row["MIC_Premium"], f"{row['MIC_Premium']:.0f}", ha='left', va='center', fontsize=12, color=colors["Premium"])

    ax.scatter(i, row["MIC_No_Premium"], color=colors["No Premium"], s=50, zorder=3)
    ax.text(i + 0.1, row["MIC_No_Premium"], f"{row['MIC_No_Premium']:.0f}", ha='left', va='center', fontsize=12, color=colors["No Premium"])

    ax.scatter(i, row["MIC_Market"], color=colors["Market"], s=50, zorder=3)
    ax.text(i + 0.1, row["MIC_Market"], f"{row['MIC_Market']:.0f}", ha='left', va='center', fontsize=12, color=colors["Market"])

    # Label
    ax.text(i, -40, f"{row['Reactor']}\n{row['Capacity']} MW", ha='center', va='top', fontsize=12)
    #ax.text(i, 0, f"{mic_df['Reactor'].iloc[i]}\n{mic_df['Capacity'].iloc[i]} MW", ha='center', va='top', fontsize=12)

# Axis formatting
ax.set_ylim(0, 500)
ax.set_xlim(-0.5, len(comparison_data) - 0.5)
ax.set_xticks([])
ax.set_ylabel("Material Input Cost [$/kW]")
ax.grid(True, linestyle='--', alpha=0.5)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color=colors["Premium"], label='MIC with Nuclear Premium', markersize=12, linestyle=''),
    Line2D([0], [0], marker='o', color=colors["No Premium"], label='MIC without Nuclear Premium', markersize=12, linestyle=''),
    Line2D([0], [0], marker='o', color=colors["Market"], label='MIC Market Price', markersize=12, linestyle='')
]
ax.legend(handles=legend_elements, loc='upper right')

# Save
os.makedirs("material_input_plots", exist_ok=True)
plt.tight_layout()
plt.savefig("material_input_plots/material_cost_range_esbwr_bwrs_updated.pdf", format="pdf")

fig, ax = plt.subplots(figsize=(8, 5))

for i in x:
    row = comparison_data.iloc[i]
    inst_vals = [row["Inst_cost_premium"], row["Inst_cost_no_premium"]]
    
    # Shaded range
    ax.fill_between([i - 0.2, i + 0.2], min(inst_vals), max(inst_vals), color='lightgray', alpha=0.4)

    # Data points
    ax.scatter(i, row["Inst_cost_premium"], color=colors["Premium"], s=50, zorder=3)
    ax.text(i + 0.1, row["Inst_cost_premium"], f"{row['Inst_cost_premium']:.0f}", ha='left', va='center', fontsize=12, color=colors["Premium"])

    ax.scatter(i, row["Inst_cost_no_premium"], color=colors["No Premium"], s=50, zorder=3)
    ax.text(i + 0.1, row["Inst_cost_no_premium"], f"{row['Inst_cost_no_premium']:.0f}", ha='left', va='center', fontsize=12, color=colors["No Premium"])

    # Label
    ax.text(i, -15, f"{row['Reactor']}\n{row['Capacity']} MW", ha='center', va='top', fontsize=12)

# Axis formatting
ax.set_ylim(0, max(comparison_data[["Inst_cost_premium", "Inst_cost_no_premium"]].max()) * 1.15)
ax.set_xlim(-0.5, len(comparison_data) - 0.5)
ax.set_xticks([])
ax.set_ylabel("Installation Cost [$/kW]")
ax.grid(True, linestyle='--', alpha=0.5)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color=colors["Premium"], label='With Nuclear Premium', markersize=12, linestyle=''),
    Line2D([0], [0], marker='o', color=colors["No Premium"], label='Without Premium', markersize=12, linestyle='')
]
ax.legend(handles=legend_elements, loc='upper right')

# Save
plt.tight_layout()
plt.savefig("material_input_plots/installation_cost_range_esbwr_bwrs_updated.pdf", format="pdf")