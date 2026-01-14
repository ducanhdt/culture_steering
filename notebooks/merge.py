import pandas as pd
import pyreadstat
import numpy as np

# --- 0. Configuration ---
# Update these paths to the actual location of your .sav files
WVS_PATH = "wvs_evs_trend/Trends_VS_1981_2022_sav_v4_0.sav"
EVS_PATH2 = "data/ZA7503_v3-0-0.dta/ZA7503_v3-0-0.dta"
OUTPUT_PATH = "wvs_evs_trend/ivs_data.pkl"

def load_spss_file(path):
    """Loads an SPSS file and returns a pandas DataFrame."""
    try:
        df, meta = pyreadstat.read_sav(path)
        return df, meta
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

# --- 1. Load Data ---
print("Loading EVS Trend File...")
evs_df, evs_meta = pyreadstat.read_dta(EVS_PATH2)
print(f"EVS dataset shape: {evs_df.shape}")
print("Loading WVS Trend File...")
wvs_df, wvs_meta = load_spss_file(WVS_PATH)
print(f"WVS dataset shape: {wvs_df.shape}")

if wvs_df is None or evs_df is None:
    raise SystemExit("Failed to load source files.")

# --- 2. Pre-processing & Standardization ---

# 2.1 Adjust types for specific columns to ensure compatibility during merge
# In SPSS: alter type studyno stdyno_w (F4.0), s016a (A6), etc.
# In Python/Pandas, we ensure they are numeric or string as needed.

cols_to_string_wvs = ['s016a', 's009', 'X048H_N1', 'X048I_N2', 'v001A_01', 
                      'v002a_01', 'x002_02B', 'versn_w', 'version', 'doi']

for col in cols_to_string_wvs:
    if col in wvs_df.columns:
        wvs_df[col] = wvs_df[col].astype(str)

# 2.2 Identification Source Flag (S001)
# 1 = WVS, 2 = EVS (Assuming standard coding; check metadata if unsure)
# The SPSS syntax implies S001 differentiates the sources. 
# If S001 is not pre-filled, you might need to set it manually:
# wvs_df['S001'] = 1
# evs_df['S001'] = 2

# --- 4. Merge Data ---
# SPSS: ADD FILES file=EVS_TF /file=WVS_TF /by=s007_01.
# This implies concatenation. 's007_01' is likely a unique ID used for sorting.

print("Merging datasets...")
# Ensure common join key exists and is same type
if 's007_01' in wvs_df.columns and 's007_01' in evs_df.columns:
    combined_df = pd.concat([evs_df, wvs_df], ignore_index=True)
    combined_df = combined_df.sort_values(by='s007_01')
else:
    print("Warning: 's007_01' missing. Concatenating without sort.")
    combined_df = pd.concat([evs_df, wvs_df], ignore_index=True)

# --- 5. Corrections (Recoding Missing Values) ---

# 5.1 Handle variables only in WVS (EVS cases set to specific missing codes)
# Logic: If S001 == 1 (WVS) is implied context, but here we fill NaN for EVS rows.
# SPSS: recode x (sysmis=-4) (else=copy). -> For variables not in EVS, fill EVS rows with -4.

# List of WVS-specific variables (from SPSS syntax section 5.1)
wvs_only_vars = [
    'A098', 'A099', 'A100', 'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 'A106B', 'A106C',
    'A124_11', 'A124_12', 'A124_14', 'A124_16', 'A124_18', 'A124_19', 'A124_30', 'A124_34',
    'A124_35', 'A124_36', 'A124_37', 'A124_38', 'A124_42', 'A124_43', 'A124_45', 'A124_46',
    'A124_47', 'A124_48', 'A124_49', 'A124_50', 'A124_51', 'A124_52', 'A189', 'A190', 'A191',
    'A192', 'A193', 'A194', 'A195', 'A196', 'A197', 'A198', 'B009', 'C008', 'C009', 'C010',
    'D055', 'D066_B', 'E025B', 'E026B', 'E069_10', 'E069_15', 'E069_21', 'E069_22', 'E069_23',
    'E069_24', 'E069_25', 'E069_26', 'E069_27', 'E069_29', 'E069_30', 'E069_31', 'E069_33',
    'E069_40', 'E069_41', 'E069_42', 'E069_43', 'E069_44', 'E069_45', 'E069_54', 'E069_55',
    'E069_56', 'E069_59', 'E113', 'E128', 'E179WVS', 'E180WVS', 'E182', 'E217', 'E218', 'E220',
    'E221B', 'E222', 'E222B', 'E234', 'E238', 'E248', 'E248B', 'E250', 'E254', 'E254B', 'E255',
    'E258B', 'E259B', 'E260B', 'E261B', 'E262B', 'E265_09', 'E266', 'F025_WVS', 'F028B',
    'F114B', 'F114C', 'F114D', 'F135A', 'F199', 'F200', 'F201', 'F202', 'F203', 'G003CS',
    'G016', 'G017', 'G018', 'G019', 'G020', 'G021', 'G022A', 'G022B', 'G022C', 'G022D', 'G022E',
    'G022F', 'G022K', 'G022L', 'G022M', 'G022N', 'G023', 'G026', 'G027', 'G027B', 'H001',
    'H002_01', 'H002_02', 'H002_03', 'H002_04', 'H002_05', 'H003_01', 'H003_02', 'H003_03',
    'H004', 'H005', 'H006_01', 'H006_02', 'H006_03', 'H006_04', 'H006_05', 'H008_01',
    'H008_02', 'H008_03', 'H008_04', 'I001', 'I002', 'X025CSWVS', 'X047_WVS', 'X047R_WVS',
    'X048ISO', 'X048WVS', 'X050C', 'X053', 'X054', 'X055'
]




# Recode NaN to -4 ("Not asked in survey") for WVS variables in non-WVS rows (EVS rows)
# Assuming S001=1 is WVS. EVS rows are where S001 != 1.
if 'S001' in combined_df.columns:
    evs_mask = combined_df['S001'] != 1
    for var in wvs_only_vars:
        if var in combined_df.columns:
            # Only fill NA values; existing values remain
            combined_df.loc[evs_mask & combined_df[var].isna(), var] = -4

    # Administrative variables recode to -3 "Not applicable"
    admin_vars_wvs = ['S002', 'S004', 'S013B']
    for var in admin_vars_wvs:
        if var in combined_df.columns:
            combined_df.loc[evs_mask & combined_df[var].isna(), var] = -3

    # Special case
    if 'S019' in combined_df.columns:
        combined_df.loc[evs_mask & combined_df['S019'].isna(), 'S019'] = 1

    # Y-variables recode to -3
    y_vars = [
        'Y003', 'Y010', 'Y011', 'Y011A', 'Y011B', 'Y011C', 'Y012', 'Y012A', 'Y012B', 'Y012C',
        'Y013', 'Y013A', 'Y013B', 'Y013C', 'Y014', 'Y014A', 'Y014B', 'Y014C', 'Y020', 'Y021',
        'Y021A', 'Y021B', 'Y021C', 'Y022', 'Y022A', 'Y022B', 'Y022C', 'Y023', 'Y023A', 'Y023B',
        'Y023C', 'Y024', 'Y024A', 'Y024B', 'Y024C'
    ]
    for var in y_vars:
        if var in combined_df.columns:
            combined_df.loc[evs_mask & combined_df[var].isna(), var] = -3

    # Alphanumeric vars
    if 'COW_ALPHA' in combined_df.columns:
        combined_df.loc[evs_mask & (combined_df['COW_ALPHA'] == ""), 'COW_ALPHA'] = "-3"
        combined_df.loc[evs_mask & combined_df['COW_ALPHA'].isna(), 'COW_ALPHA'] = "-3"


# 5.2 Handle variables only in EVS (WVS cases set to specific missing codes)
# Logic: If S001=2 (EVS) is implied context, but here we fill NaN for WVS rows.
# WVS rows are where S001 != 2 (or S001 == 1)

    wvs_mask = combined_df['S001'] != 2
    evs_only_vars = [
        'A043_01', 'A124_26', 'C029', 'D020', 'D026', 'D039', 'D043_01', 'D064', 'E042', 'E069_16',
        'E144', 'E151', 'E152', 'E153', 'E154', 'E155', 'E156', 'E157', 'E158', 'E159', 'E160',
        'E161', 'E162', 'E179', 'E181', 'E181A', 'E181C', 'E197', 'F025_EVS', 'F026', 'F030',
        'F099', 'F114', 'F131', 'F137', 'F138', 'F144_01', 'G014', 'G033', 'G034', 'G035', 'G036',
        'G038', 'G040', 'G041', 'G043', 'G051', 'V004AF', 'V004DF', 'V004EF', 'V005F', 'V011',
        'V012', 'V013', 'V014', 'V015', 'V016', 'V017', 'V018', 'W001', 'W001A', 'W002A', 'W002E',
        'W004', 'W005_2', 'W005_2_01', 'W006A', 'W006B', 'W006C', 'W006D', 'W007', 'W008', 'W009',
        'X002_02', 'X002_03', 'X004', 'X006', 'X007_01', 'X007_02', 'X025A', 'X025CSEVS', 'X028_01',
        'X032R_01', 'X034R_01', 'X035_2', 'X035_2_01', 'X036A', 'X036B', 'X036C', 'X036D', 'X037_01',
        'X037_02', 'X047_EVS', 'X047D', 'X047R_EVS', 'X048_EVS'
    ]

    for var in evs_only_vars:
        if var in combined_df.columns:
            combined_df.loc[wvs_mask & combined_df[var].isna(), var] = -4

    # Administrative/Technical EVS vars
    evs_admin_vars = [
        'mm_fw_end_fu_EVS5', 'mm_fw_start_fu_EVS5', 'mm_matrix_group_EVS5', 'mm_mixed_mode_EVS5',
        'mm_mode_fu_EVS5', 'mm_v277_fu_EVS5', 'mm_v278a_fu_r_EVS5', 'mm_v279a_fu_r_EVS5',
        'mm_year_fu_EVS5', 'S002evs', 'S036'
    ]
    for var in evs_admin_vars:
        if var in combined_df.columns:
            combined_df.loc[wvs_mask & combined_df[var].isna(), var] = -3

    # Alphanumeric EVS vars
    evs_str_vars = ['W001A_01', 'X048b_n2', 'X048a_n1']
    for var in evs_str_vars:
        if var in combined_df.columns:
            combined_df.loc[wvs_mask & combined_df[var].isna(), var] = "-4"
            combined_df.loc[wvs_mask & (combined_df[var] == ""), var] = "-4"

# 5.3 Specific Recodes
# recode X048H_N1 ("     -1"="-1").
if 'X048H_N1' in combined_df.columns:
    combined_df['X048H_N1'] = combined_df['X048H_N1'].replace("     -1", "-1")
if 'X048I_N2' in combined_df.columns:
    combined_df['X048I_N2'] = combined_df['X048I_N2'].replace("     -1", "-1")

# recode V001A_01 V002A_01 W001A_01 X002_02B ("OTH" = "ZZZ").
replace_oth_vars = ['V001A_01', 'V002A_01', 'W001A_01', 'X002_02B']
for var in replace_oth_vars:
    if var in combined_df.columns:
        combined_df[var] = combined_df[var].replace("OTH", "ZZZ")

final_df = combined_df

print(f"Final dataset shape after keeping specified variables: {final_df.shape}")
print(f"Saving merged file to {OUTPUT_PATH}...")
final_df.to_pickle(OUTPUT_PATH)



meta_col = ["S020", "S003"]
# Weights
weights = ["S017"]
# Use the ten questions from the IVS that form the basis of the Inglehart-Welzel Cultural Map
iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]
y003_qns = ['A040', 'A042', 'A029', 'A039']
ivs_data = final_df[meta_col+weights+iv_qns+y003_qns]
ivs_data = ivs_data.rename(columns={'S020': 'year', 'S003': 's003', 'S017': 'weight'})
ivs_data['Y003'] = np.where(
	(ivs_data['A040'] >= 0) & (ivs_data['A042'] >= 0) & 
	(ivs_data['A029'] >= 0) & (ivs_data['A039'] >= 0),
	(ivs_data['A029'] + ivs_data['A039']) - (ivs_data['A040'] + ivs_data['A042']),
	-5
)
# drop the intermediate columns
ivs_data = ivs_data.drop(columns=['A040', 'A042', 'A029', 'A039'])

ivs_data = ivs_data[ivs_data["year"] >= 2005]
for col in ['A008', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']:
	ivs_data[col] = ivs_data[col].where(ivs_data[col] > 0, np.nan)
ivs_data['Y003'] = ivs_data['Y003'].where(ivs_data['Y003'] > -5, np.nan)
#drop rows with all iv_qns as nan
ivs_data = ivs_data.dropna(subset=iv_qns, how='any')


print(f"IVS countries: {ivs_data['s003'].nunique()}")
# save processed ivs_data to pkl
ivs_data.to_pickle("../wvs_evs_trend/ivs_data_processed.pkl")
