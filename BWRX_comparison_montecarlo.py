import numpy as np
import pickle
from atb_data import values_from_ATB2024
from tqdm import tqdm
import pandas as pd


np.random.seed(42)
years = [2030, 2035, 2040, 2045, 2050]
num_simulations = 10000

#OCC SIMULATION
with open("comparison_data.pkl", "rb") as f:
    comparison_data = pickle.load(f)

mic_nuclear_premium,mic_nuclear_no_premium, mic_nuclear_market_price = comparison_data.loc[comparison_data["Reactor"] == "ESBWR", "MIC_Premium"].values[0], comparison_data.loc[comparison_data["Reactor"] == "ESBWR", "MIC_No_Premium"].values[0], comparison_data.loc[comparison_data["Reactor"] == "ESBWR", "MIC_Market"].values[0]
mic_smr_premium, mic_smr_no_premium, mic_smr_market_price = comparison_data.loc[comparison_data["Reactor"] == "BWRX", "MIC_Premium"].values[0], comparison_data.loc[comparison_data["Reactor"] == "BWRX", "MIC_No_Premium"].values[0], comparison_data.loc[comparison_data["Reactor"] == "BWRX", "MIC_Market"].values[0]

nuclear_occ_range = list(values_from_ATB2024["OCC + GCC"][:3])
smr_occ_range = list(values_from_ATB2024["OCC + GCC"][3:6])
def sample_lr_nuclear():
    return np.random.triangular(0.05, 0.10, 0.15)

def compute_b(lr):
    return np.log(1 - lr) / np.log(2)

def sample_deployment_triangular(q3, q2, q1):
    if np.isclose(q3, q1):
        return q3
    else:
        return np.random.triangular(left=q3, mode=q2, right=q1)

results_noak_simulation = {f"Deployment {year}": {
    "nuc_noak_fixed_premium": [],
    "nuc_noak_dynamic_premium": [],
    "nuc_noak_fixed_no_premium": [],
    "nuc_noak_dynamic_no_premium": [],
    "nuc_noak_fixed_market_price": [],
    "nuc_noak_dynamic_market_price": [],
    "smr_noak_fixed_premium": [],
    "smr_noak_dynamic_premium": [],
    "smr_noak_fixed_no_premium": [],
    "smr_noak_dynamic_no_premium": [],
    "smr_noak_fixed_market_price": [],
    "smr_noak_dynamic_market_price": []
} for year in years}

for sim in tqdm(range(num_simulations), desc="Simulation Progress"):
    occ_initial_nuclear = np.random.triangular(min(nuclear_occ_range), nuclear_occ_range[1], max(nuclear_occ_range))
    occ_initial_smr = np.random.triangular(min(smr_occ_range), smr_occ_range[1], max(smr_occ_range))

    nuclear_mic_premium = np.random.normal(mic_nuclear_premium, 0.35 * mic_nuclear_premium)
    nuclear_mic_no_premium = np.random.normal(mic_nuclear_no_premium, 0.35 * mic_nuclear_no_premium)
    nuclear_mic_market_price = np.random.normal(mic_nuclear_market_price, 0.35 * mic_nuclear_market_price)
    smr_mic_premium = np.random.normal(mic_smr_premium, 0.35 * mic_smr_premium)
    smr_mic_no_premium = np.random.normal(mic_smr_no_premium, 0.35 * mic_smr_no_premium)
    smr_mic_market_price = np.random.normal(mic_smr_market_price, 0.35 * mic_smr_market_price)

    lr_nuclear = sample_lr_nuclear()
    b_nuclear = compute_b(lr_nuclear)
    lr_smr = sample_lr_nuclear()
    b_smr = compute_b(lr_smr)

    deployments_nuclear = {}
    deployments_smr = {}
    prev_deployment_nuclear = 1
    prev_deployment_smr = 1

    for year in years:
        nuclear_adv= values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Adv", f"Deployment {year}"].values[0]
        nuclear_mod= values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Mod", f"Deployment {year}"].values[0]
        nuclear_cons= values_from_ATB2024.loc[values_from_ATB2024["Category"] == "Nuclear-Cons", f"Deployment {year}"].values[0]
        smr_adv= values_from_ATB2024.loc[values_from_ATB2024["Category"] == "SMR-Adv", f"Deployment {year}"].values[0]
        smr_mod= values_from_ATB2024.loc[values_from_ATB2024["Category"] == "SMR-Mod", f"Deployment {year}"].values[0]
        smr_cons= values_from_ATB2024.loc[values_from_ATB2024["Category"] == "SMR-Cons", f"Deployment {year}"].values[0]

        while True:
            sampled = sample_deployment_triangular(nuclear_cons, nuclear_mod, nuclear_adv)
            if sampled >= prev_deployment_nuclear:
                deployments_nuclear[year] = sampled
                prev_deployment_nuclear = sampled
                break
        while True:
            sampled = sample_deployment_triangular(smr_cons, smr_mod, smr_adv)
            if sampled >= prev_deployment_smr:
                deployments_smr[year] = sampled
                prev_deployment_smr = sampled
                break
    
    for mic_name, (nuc_mic_value, smr_mic_value) in zip(
        ["premium", "no_premium", "market_price"],
        [
            (nuclear_mic_premium, smr_mic_premium),
            (nuclear_mic_no_premium, smr_mic_no_premium),
            (nuclear_mic_market_price, smr_mic_market_price),
        ]
    ):
        # Nuclear
        overhead_fixed_nuc = occ_initial_nuclear - nuc_mic_value
        overhead_dynamic_nuc = occ_initial_nuclear - nuc_mic_value
        occ_prev_fixed_nuc = occ_initial_nuclear
        occ_prev_dynamic_nuc = occ_initial_nuclear
        N_prev_nuc = 1

        # SMR
        overhead_fixed_smr = occ_initial_smr - smr_mic_value
        overhead_dynamic_smr = occ_initial_smr - smr_mic_value
        occ_prev_fixed_smr = occ_initial_smr
        occ_prev_dynamic_smr = occ_initial_smr
        N_prev_smr = 1

        for year in years:
            N_curr_nuc = deployments_nuclear[year]
            N_curr_smr = deployments_smr[year]

            # --- Fixed Learning: Nuclear ---
            overhead_fixed_new_nuc = overhead_fixed_nuc * (N_curr_nuc / N_prev_nuc) ** b_nuclear
            noak_fixed_nuc = nuc_mic_value + overhead_fixed_new_nuc
            results_noak_simulation[f"Deployment {year}"][f"nuc_noak_fixed_{mic_name}"].append(noak_fixed_nuc)

            # --- Dynamic Learning: Nuclear ---
            overhead_ratio_nuc = overhead_dynamic_nuc / occ_prev_dynamic_nuc
            lr_dynamic_nuc = min(0.99, lr_nuclear * (1 + overhead_ratio_nuc))
            b_dynamic_nuc = compute_b(lr_dynamic_nuc)
            overhead_dynamic_new_nuc = overhead_dynamic_nuc * (N_curr_nuc / N_prev_nuc) ** b_dynamic_nuc
            noak_dynamic_nuc = nuc_mic_value + overhead_dynamic_new_nuc
            results_noak_simulation[f"Deployment {year}"][f"nuc_noak_dynamic_{mic_name}"].append(noak_dynamic_nuc)

            # --- Fixed Learning: SMR ---
            overhead_fixed_new_smr = overhead_fixed_smr * (N_curr_smr / N_prev_smr) ** b_smr
            noak_fixed_smr = smr_mic_value + overhead_fixed_new_smr
            results_noak_simulation[f"Deployment {year}"][f"smr_noak_fixed_{mic_name}"].append(noak_fixed_smr)

            # --- Dynamic Learning: SMR ---
            overhead_ratio_smr = overhead_dynamic_smr / occ_prev_dynamic_smr
            lr_dynamic_smr = min(0.99, lr_smr * (1 + overhead_ratio_smr))
            b_dynamic_smr = compute_b(lr_dynamic_smr)
            overhead_dynamic_new_smr = overhead_dynamic_smr * (N_curr_smr / N_prev_smr) ** b_dynamic_smr
            noak_dynamic_smr = smr_mic_value + overhead_dynamic_new_smr
            results_noak_simulation[f"Deployment {year}"][f"smr_noak_dynamic_{mic_name}"].append(noak_dynamic_smr)

            # Update states
            occ_prev_fixed_nuc = noak_fixed_nuc
            occ_prev_dynamic_nuc = noak_dynamic_nuc
            overhead_fixed_nuc = overhead_fixed_new_nuc
            overhead_dynamic_nuc = overhead_dynamic_new_nuc
            N_prev_nuc = N_curr_nuc

            occ_prev_fixed_smr = noak_fixed_smr
            occ_prev_dynamic_smr = noak_dynamic_smr
            overhead_fixed_smr = overhead_fixed_new_smr
            overhead_dynamic_smr = overhead_dynamic_new_smr
            N_prev_smr = N_curr_smr

with open("noak_simulation_results_nuclear_smr.pkl", "wb") as f:
    pickle.dump(results_noak_simulation, f)

print("NOAK simulation completed and saved.")



#LCOE simulation
WACC_nuc = values_from_ATB2024.loc[1, "WACC"]
WACC_smr = WACC_nuc
CRP_nuc = values_from_ATB2024.loc[1, "CRP"]
CRP_smr=CRP_nuc
PFF_nuc = values_from_ATB2024.loc[1, "PFF"]
PFF_smr= PFF_nuc

# Define triangular input ranges
CFF_range_nuclear = list(values_from_ATB2024["CFF"][:3])
CFF_range_smr = list(values_from_ATB2024["CFF"][3:6])
FOM_range_nuclear = list(values_from_ATB2024["FOM"][:3])
FOM_range_smr = list(values_from_ATB2024["FOM"][3:6])
CF_nuclear = values_from_ATB2024.loc[1, "CF"]
CF_smr = values_from_ATB2024.loc[3, "CF"]
VOM_range_nuclear = list(values_from_ATB2024["VOM"][:3])
VOM_range_smr = list(values_from_ATB2024["VOM"][3:6])
Fuel_range_nuclear = list(values_from_ATB2024["Fuel"][:3])
Fuel_range_smr = list(values_from_ATB2024["Fuel"][3:6])

# --- Helper functions ---
def calc_crf(wacc, crp):
    return wacc / (1 - (1 / (1 + wacc) ** crp))

def calc_lcoe(crf, pff, cff, occ, fom, cf, vom=0, fuel=0):
    return ((crf * pff * cff * occ + fom) * 1000 / (cf * 8760)) + vom + fuel

lcoe_results = {
    f"Deployment {year}": {
        "nuc_lcoe_fixed_premium": [],
        "nuc_lcoe_dynamic_premium": [],
        "nuc_lcoe_fixed_no_premium": [],
        "nuc_lcoe_dynamic_no_premium": [],
        "nuc_lcoe_fixed_market_price": [],
        "nuc_lcoe_dynamic_market_price": [],
        "smr_lcoe_fixed_premium": [],
        "smr_lcoe_dynamic_premium": [],
        "smr_lcoe_fixed_no_premium": [],
        "smr_lcoe_dynamic_no_premium": [],
        "smr_lcoe_fixed_market_price": [],
        "smr_lcoe_dynamic_market_price": []
    }
    for year in years
}

for year in years:
    year_label = f"Deployment {year}"
    for i in range(num_simulations):
        # --- Sample uncertain inputs ---
        cff_nuc = np.random.triangular(*CFF_range_nuclear)
        cff_smr = np.random.triangular(*CFF_range_smr)
        fom_nuc = np.random.triangular(*FOM_range_nuclear)
        fom_smr = np.random.triangular(*FOM_range_smr)
        vom_nuc = np.random.triangular(*VOM_range_nuclear)
        vom_smr = np.random.triangular(*VOM_range_smr)
        fuel_nuc = np.random.triangular(*Fuel_range_nuclear)
        fuel_smr = np.random.triangular(*Fuel_range_smr)

        crf_nuc = calc_crf(WACC_nuc, CRP_nuc)
        crf_smr = calc_crf(WACC_smr, CRP_smr)

        for scenario in ["premium", "no_premium", "market_price"]:
            occ_fixed_nuc = results_noak_simulation[year_label][f"nuc_noak_fixed_{scenario}"][i]
            occ_dynamic_nuc = results_noak_simulation[year_label][f"nuc_noak_dynamic_{scenario}"][i]
            occ_fixed_smr = results_noak_simulation[year_label][f"smr_noak_fixed_{scenario}"][i]
            occ_dynamic_smr = results_noak_simulation[year_label][f"smr_noak_dynamic_{scenario}"][i]

            lcoe_fixed_nuc = calc_lcoe(crf_nuc, PFF_nuc, cff_nuc, occ_fixed_nuc, fom_nuc, CF_nuclear, vom=vom_nuc, fuel=fuel_nuc)
            lcoe_dynamic_nuc = calc_lcoe(crf_nuc, PFF_nuc, cff_nuc, occ_dynamic_nuc, fom_nuc, CF_nuclear, vom=vom_nuc, fuel=fuel_nuc)
            lcoe_fixed_smr = calc_lcoe(crf_smr, PFF_smr, cff_smr, occ_fixed_smr, fom_smr, CF_smr, vom=vom_smr, fuel=fuel_smr)
            lcoe_dynamic_smr = calc_lcoe(crf_smr, PFF_smr, cff_smr, occ_dynamic_smr, fom_smr, CF_smr, vom=vom_smr, fuel=fuel_smr)

            lcoe_results[year_label][f"nuc_lcoe_fixed_{scenario}"].append(lcoe_fixed_nuc)
            lcoe_results[year_label][f"nuc_lcoe_dynamic_{scenario}"].append(lcoe_dynamic_nuc)
            lcoe_results[year_label][f"smr_lcoe_fixed_{scenario}"].append(lcoe_fixed_smr)
            lcoe_results[year_label][f"smr_lcoe_dynamic_{scenario}"].append(lcoe_dynamic_smr)


with open("lcoe_simulation_results_nuclear_smr.pkl", "wb") as f:
    pickle.dump(lcoe_results, f)
print("LCOE simulation completed and saved.")
