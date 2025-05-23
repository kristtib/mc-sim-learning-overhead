import pandas as pd

values_from_ATB2024 = pd.DataFrame({
    "Category": ["Nuclear-Adv", "Nuclear-Mod", "Nuclear-Cons", "SMR-Adv", "SMR-Mod", "SMR-Cons", "FOW-Class11-Adv", "FOW_Class11-Mod", "FOW_Class11-Cons"],
    "Deployment 2030": [1, 1, 1, 3, 3, 1, 6.4, 4, 1.6],
    "Deployment 2035": [14, 3, 1, 42, 9, 1, 32, 20, 8],
    "Deployment 2040": [58, 8.5, 3, 174, 25.5, 9, 75, 50, 16],
    "Deployment 2045": [124, 17, 6, 372, 51, 18, 120, 80, 24],
    "Deployment 2050": [200, 34, 12, 600, 102, 36, 160, 110, 32],
    "OCC" : [5722, 6267.5, 8447.5, 5995, 8720, 10900, 4586, 6648, 10404],
    "OCC + GCC" : [5831, 6376.5, 8556.5, 6104, 8829, 11009, 9216, 11317, 14754],
    "WACC" : [0.047, 0.047, 0.047, 0.047, 0.047, 0.047, 0.04, 0.04, 0.04],
    "CRP" : [60,60,60,60,60,60,30,30,30],
    "PFF" : [1.061, 1.061, 1.061, 1.061, 1.061, 1.061, 1.056, 1.056, 1.056],
    "CFF" : [1.210, 1.302, 1.508, 1.146, 1.191, 1.256, 1.109, 1.109, 1.109],
    "CapRegMult" : [1,1,1,1,1,1,1,1,1],
    "FOM" : [133.56, 185.5, 216.24, 125.1, 148.24, 235.44, 69, 75.3, 83.4],
    "VOM" : [2.07, 3.05, 3.7, 2.4, 2.83, 3.05, 0 ,0 ,0],
    "Fuel" : [9.8, 10.9, 12, 10.9, 12, 13.08, 0,0,0],
    "CF" : [0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.5, 0.48, 0.46],
    "learning_rate" : [0.08, 0.08, 0.08, 0.095, 0.095, 0.095, 0.142, 0.115, 0.087],
})
