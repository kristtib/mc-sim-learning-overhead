import numpy as np
import pickle
from atb_data import values_from_ATB2024


with open("noak_simulation_results_fixed_dynamic.pkl", "rb") as f:
    noak_results = pickle.load(f)

# Define simulation parameters
years = [2030, 2035, 2040, 2045, 2050]
num_simulations = 10000 

# --- Extract fixed constants ---
WACC_nuc = values_from_ATB2024.loc[1, "WACC"]
WACC_wind = values_from_ATB2024.loc[7, "WACC"]
CRP_nuc = values_from_ATB2024.loc[1, "CRP"]
CRP_wind = values_from_ATB2024.loc[7, "CRP"]
PFF_nuc = values_from_ATB2024.loc[1, "PFF"]
PFF_wind = values_from_ATB2024.loc[7, "PFF"]

# Define triangular input ranges
CFF_range_nuclear = list(values_from_ATB2024["CFF"][:3])
CFF_wind = values_from_ATB2024.loc[7, "CFF"]
FOM_range_nuclear = list(values_from_ATB2024["FOM"][:3])
FOM_range_wind = list(values_from_ATB2024["FOM"][6:])
CF_nuclear = values_from_ATB2024.loc[1, "CF"]
CF_range_wind = list(values_from_ATB2024["CF"][6:])
VOM_range_nuclear = list(values_from_ATB2024["VOM"][:3])
Fuel_range_nuclear = list(values_from_ATB2024["Fuel"][:3])

# --- Helper functions ---
def calc_crf(wacc, crp):
    return wacc / (1 - (1 / (1 + wacc) ** crp))

def calc_lcoe(crf, pff, cff, occ, fom, cf, vom=0, fuel=0):
    return ((crf * pff * cff * occ + fom) * 1000 / (cf * 8760)) + vom + fuel


lcoe_results = {
    f"Deployment {year}": {
        "fixed": {
            "premium": [],
            "no_premium": [],
            "market_price": [],
            "wind": []
        },
        "dynamic": {
            "premium": [],
            "no_premium": [],
            "market_price": [],
            "wind": []
        }
    }
    for year in years
}

# --- Perform LCOE simulation ---
for year in years:
    year_label = f"Deployment {year}"
    for i in range(num_simulations):
        # --- Sample uncertain inputs ---
        cff_nuc = np.random.triangular(*CFF_range_nuclear)
        fom_nuc = np.random.triangular(*FOM_range_nuclear)
        fom_wind = np.random.triangular(*FOM_range_wind)
        vom_nuc = np.random.triangular(*VOM_range_nuclear)
        fuel_nuc = np.random.triangular(*Fuel_range_nuclear)
        cf_wind = np.random.triangular(CF_range_wind[2], CF_range_wind[1], CF_range_wind[0])

        # --- Compute CRFs ---
        crf_nuc = calc_crf(WACC_nuc, CRP_nuc)
        crf_wind = calc_crf(WACC_wind, CRP_wind)

        # --- Nuclear Scenarios ---
        for scenario in ["premium", "no_premium", "market_price"]:
            occ_fixed = noak_results[year_label][f"noak_fixed_{scenario}"][i]
            occ_dyn = noak_results[year_label][f"noak_dynamic_{scenario}"][i]

            lcoe_fixed = calc_lcoe(crf_nuc, PFF_nuc, cff_nuc, occ_fixed, fom_nuc, CF_nuclear, vom_nuc, fuel_nuc)
            lcoe_dyn = calc_lcoe(crf_nuc, PFF_nuc, cff_nuc, occ_dyn, fom_nuc, CF_nuclear, vom_nuc, fuel_nuc)

            lcoe_results[year_label]["fixed"][scenario].append(lcoe_fixed)
            lcoe_results[year_label]["dynamic"][scenario].append(lcoe_dyn)

        # --- Wind Scenarios ---
        occ_wind_fixed = noak_results[year_label]["noak_wind_fixed"][i]
        occ_wind_dyn = noak_results[year_label]["noak_wind_dynamic"][i]

        lcoe_wind_fixed = calc_lcoe(crf_wind, PFF_wind, CFF_wind, occ_wind_fixed, fom_wind, cf_wind)
        lcoe_wind_dyn = calc_lcoe(crf_wind, PFF_wind, CFF_wind, occ_wind_dyn, fom_wind, cf_wind)

        lcoe_results[year_label]["fixed"]["wind"].append(lcoe_wind_fixed)
        lcoe_results[year_label]["dynamic"]["wind"].append(lcoe_wind_dyn)


with open("lcoe_simulation_results_fixed_dynamic.pkl", "wb") as f:
    pickle.dump(lcoe_results, f)

print("LCOE simulation completed and saved.")
