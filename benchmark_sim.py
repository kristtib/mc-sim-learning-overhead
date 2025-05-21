import numpy as np
import matplotlib.pyplot as plt
from atb_data import values_from_ATB2024
import pickle 
from tqdm import tqdm

np.random.seed(42)
num_simulations = 10000

years = ["2030", "2035", "2040", "2045", "2050"]

nuclear_occ_range = list(values_from_ATB2024["OCC + GCC"][:3])
smr_occ_range = list(values_from_ATB2024["OCC + GCC"][3:6])
wind_occ_range = list(values_from_ATB2024["OCC + GCC"][6:])

def sample_lr_nuclear():
    return np.random.triangular(0.05, 0.10, 0.15)

def sample_lr_wind():
    return np.random.beta(5, 10) * 0.3

def compute_b(lr):
    return np.log(1 - lr) / np.log(2)

def sample_deployment_triangular(q3, q2, q1):
    if np.isclose(q3, q1):
        return q3
    else:
        return np.random.triangular(left=q3, mode=q2, right=q1)

# --- Prepare Results Dictionary ---
results = {year: {
    "noak_nuclear": [],
    "noak_smr":[],
    "noak_wind": [],
} for year in years}

# --- Run Monte Carlo Simulations ---
for sim in tqdm(range(num_simulations), desc="Running simulations"):

    # --- Sample Initial OCC ---
    occ_initial_nuclear = np.random.triangular(min(nuclear_occ_range), nuclear_occ_range[1], max(nuclear_occ_range))
    occ_initial_smr = np.random.triangular(min(smr_occ_range),smr_occ_range[1],max(smr_occ_range))
    occ_initial_wind = np.random.triangular(min(wind_occ_range), wind_occ_range[1], max(wind_occ_range))

    # --- Sample Learning Rates ---
    lr_nuclear = sample_lr_nuclear()
    lr_wind = sample_lr_wind()
    b_fixed_nuclear = compute_b(lr_nuclear)
    b_fixed_wind = compute_b(lr_wind)

    # --- Sample Deployments for this run ---
    deployments_nuclear = {}
    deployments_smr={}
    deployments_wind = {}
    prev_deployment_nuclear = 1
    prev_deployment_smr = 1
    prev_deployment_wind = 1

    for year in years:
        # Nuclear
        nuclear_adv = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Adv", f"Deployment {year}"].values[0]
        nuclear_mod = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Mod", f"Deployment {year}"].values[0]
        nuclear_cons = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Cons", f"Deployment {year}"].values[0]

        while True:
            sampled = sample_deployment_triangular(nuclear_cons, nuclear_mod, nuclear_adv)
            if sampled >= prev_deployment_nuclear:
                deployments_nuclear[year] = sampled
                prev_deployment_nuclear = sampled
                break
        # smr
        smr_adv = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "SMR-Adv", f"Deployment {year}"].values[0]
        smr_mod = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "SMR-Mod", f"Deployment {year}"].values[0]
        smr_cons = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "SMR-Cons", f"Deployment {year}"].values[0]

        while True:
            sampled = sample_deployment_triangular(smr_cons, smr_mod, smr_adv)
            if sampled >= prev_deployment_smr:
                deployments_smr[year] = sampled
                prev_deployment_smr = sampled
                break
        # Wind
        wind_adv = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "FOW-Class11-Adv", f"Deployment {year}"].values[0]
        wind_mod = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "FOW_Class11-Mod", f"Deployment {year}"].values[0]
        wind_cons = values_from_ATB2024.loc[values_from_ATB2024["Category"] == "FOW_Class11-Cons", f"Deployment {year}"].values[0]

        while True:
            sampled = sample_deployment_triangular(wind_cons, wind_mod, wind_adv)
            if sampled >= prev_deployment_wind:
                deployments_wind[year] = sampled
                prev_deployment_wind = sampled
                break
        # --- Compute NOAK Costs for each year ---
    for year in years:
        # Calculate NOAK cost using learning curve
        cumulative_deployment_nuclear = deployments_nuclear[year]
        cumulative_deployment_smr = deployments_smr[year]
        cumulative_deployment_wind = deployments_wind[year]

        noak_nuclear = occ_initial_nuclear * (cumulative_deployment_nuclear ** b_fixed_nuclear)
        noak_smr = occ_initial_smr * (cumulative_deployment_smr**b_fixed_nuclear)
        noak_wind = occ_initial_wind * (cumulative_deployment_wind ** b_fixed_wind)

        results[year]["noak_nuclear"].append(noak_nuclear)
        results[year]["noak_smr"].append(noak_smr)
        results[year]["noak_wind"].append(noak_wind)

# ---Save results ---
with open("benchmark_simulation_results.pkl", "wb") as f:
    pickle.dump(results, f)
#################################################################################33
#LCOE benchmark simulation
years = [2030, 2035, 2040, 2045, 2050]
num_simulations = 10000  # Must match original NOAK simulation

WACC_nuc = values_from_ATB2024.loc[1, "WACC"]
WACC_smr = WACC_nuc
WACC_wind = values_from_ATB2024.loc[7, "WACC"]
CRP_nuc = values_from_ATB2024.loc[1, "CRP"]
CRP_smr=CRP_nuc
CRP_wind = values_from_ATB2024.loc[7, "CRP"]
PFF_nuc = values_from_ATB2024.loc[1, "PFF"]
PFF_smr= PFF_nuc
PFF_wind = values_from_ATB2024.loc[7, "PFF"]

# Define triangular input ranges
CFF_range_nuclear = list(values_from_ATB2024["CFF"][:3])
CFF_range_smr = list(values_from_ATB2024["CFF"][3:6])
CFF_wind = values_from_ATB2024.loc[7, "CFF"]
FOM_range_nuclear = list(values_from_ATB2024["FOM"][:3])
FOM_range_smr = list(values_from_ATB2024["FOM"][3:6])
FOM_range_wind = list(values_from_ATB2024["FOM"][6:])
CF_nuclear = values_from_ATB2024.loc[1, "CF"]
CF_smr = values_from_ATB2024.loc[3, "CF"]
CF_range_wind = list(values_from_ATB2024["CF"][6:])
VOM_range_nuclear = list(values_from_ATB2024["VOM"][:3])
VOM_range_smr = list(values_from_ATB2024["VOM"][3:6])
Fuel_range_nuclear = list(values_from_ATB2024["Fuel"][:3])
Fuel_range_smr = list(values_from_ATB2024["Fuel"][3:6])

def calc_crf(wacc, crp):
    return wacc / (1 - (1 / (1 + wacc) ** crp))

def calc_lcoe(crf, pff, cff, occ, fom, cf, vom=0, fuel=0):
    return ((crf * pff * cff * occ + fom) * 1000 / (cf * 8760)) + vom + fuel
lcoe_results = {
    f"Deployment {year}": {
        "lcoe_nuc": [],
        "lcoe_smr": [],
        "lcoe_wind": []
    }
    for year in years
}

# --- Compute LCOE for each technology ---
for year in years:
    for i in range(num_simulations):
        # --- Sample uncertain inputs ---
        cff_nuc = np.random.triangular(*CFF_range_nuclear)
        cff_smr = np.random.triangular(*CFF_range_smr)
        fom_nuc = np.random.triangular(*FOM_range_nuclear)
        fom_smr = np.random.triangular(*FOM_range_smr)
        fom_wind = np.random.triangular(*FOM_range_wind)
        vom_nuc = np.random.triangular(*VOM_range_nuclear)
        vom_smr = np.random.triangular(*VOM_range_smr)
        fuel_nuc = np.random.triangular(*Fuel_range_nuclear)
        fuel_smr = np.random.triangular(*Fuel_range_smr)
        cf_wind = np.random.triangular(CF_range_wind[2], CF_range_wind[1], CF_range_wind[0])

        # --- Calculate CRFs ---
        crf_nuc = calc_crf(WACC_nuc, CRP_nuc)
        crf_smr = calc_crf(WACC_smr, CRP_smr)
        crf_wind = calc_crf(WACC_wind, CRP_wind)

        # --- Retrieve OCC values from benchmark results ---
        occ_nuc = results[str(year)]["noak_nuclear"][i]
        occ_smr = results[str(year)]["noak_smr"][i]
        occ_wind = results[str(year)]["noak_wind"][i]

        # --- Compute LCOEs (excluding fuel in wind) ---
        lcoe_nuc = calc_lcoe(crf_nuc, PFF_nuc, cff_nuc, occ_nuc, fom_nuc, CF_nuclear, vom=vom_nuc, fuel=fuel_nuc)
        lcoe_smr = calc_lcoe(crf_smr, PFF_smr, cff_smr, occ_smr, fom_smr, CF_smr, vom=vom_smr, fuel=fuel_smr)
        lcoe_wind = calc_lcoe(crf_wind, PFF_wind, CFF_wind, occ_wind, fom_wind, cf_wind)

        # --- Store results ---
        lcoe_results[f"Deployment {year}"]["lcoe_nuc"].append(lcoe_nuc)
        lcoe_results[f"Deployment {year}"]["lcoe_smr"].append(lcoe_smr)
        lcoe_results[f"Deployment {year}"]["lcoe_wind"].append(lcoe_wind)

# --- Save results ---
with open("benchmark_lcoe_simulation_results.pkl", "wb") as f:
    pickle.dump(lcoe_results, f)