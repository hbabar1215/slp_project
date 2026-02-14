import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("data/clean_output/df.csv")
print("Original data shape:", df.shape)

# Safely select Likert columns Q21–Q52
likert_cols = []
for col in df.columns:
    if col.startswith("Q"):
        # remove Q and take only the first number before "_" if present
        number_part = col[1:].split("_")[0]
        try:
            number = int(number_part)
            if 21 <= number <= 52:
                likert_cols.append(col)
        except:
            continue

print("Likert columns selected:", likert_cols)

# Split complete vs incomplete surveys
df_complete = df.dropna(subset=likert_cols).copy()    # complete = no missing in Likert
df_incomplete = df[df[likert_cols].isna().any(axis=1)].copy()  # incomplete = some missing

print("Complete surveys:", df_complete.shape)
print("Incomplete surveys:", df_incomplete.shape)

# calculate mean HD_Familiarity for Q21, Q28–Q31
hd_fam_cols = ["Q21","Q28","Q29","Q30","Q31"]
hd_fam_cols = [c for c in hd_fam_cols if c in df_complete.columns]  # safe in case some missing
df_complete["HD_Familiarity"] = df_complete[hd_fam_cols].mean(axis=1)
df_incomplete["HD_Familiarity"] = df_incomplete[hd_fam_cols].mean(axis=1)

# Graduate curriculum coverage
grad_curr_cols = ["Q32","Q33"]
grad_curr_cols = [c for c in grad_curr_cols if c in df_complete.columns]
df_complete["Grad_Curriculum"] = df_complete[grad_curr_cols].mean(axis=1)
df_incomplete["Grad_Curriculum"] = df_incomplete[grad_curr_cols].mean(axis=1)

# Importance of HD care (Q22, Q23_1-3, Q24_1-2, Q25)
importance_cols = ["Q22","Q23_1","Q23_2","Q23_3","Q24_1","Q24_2","Q25"]
importance_cols = [c for c in importance_cols if c in df_complete.columns]
df_complete["Importance_HD_Care"] = df_complete[importance_cols].mean(axis=1)
df_incomplete["Importance_HD_Care"] = df_incomplete[importance_cols].mean(axis=1)

# HD Exposure / clinical experience (Q26_1-6, Q27_1-6, Q50)
exposure_cols = ["Q26_1","Q26_2","Q26_3","Q26_4","Q26_5","Q26_6",
                 "Q27_1","Q27_2","Q27_3","Q27_4","Q27_5","Q27_6","Q50"]
exposure_cols = [c for c in exposure_cols if c in df_complete.columns]
df_complete["HD_Exposure"] = df_complete[exposure_cols].mean(axis=1)
df_incomplete["HD_Exposure"] = df_incomplete[exposure_cols].mean(axis=1)

# Preparedness / confidence (Q41–Q44, Q47_1–5, Q48_1–2, Q51–Q52_8)
prepared_cols = ["Q41","Q42","Q43","Q44","Q47_1","Q47_2","Q47_3","Q47_4","Q47_5",
                 "Q48_1","Q48_2","Q51","Q52_1","Q52_2","Q52_3","Q52_4","Q52_5",
                 "Q52_6","Q52_7","Q52_8"]
prepared_cols = [c for c in prepared_cols if c in df_complete.columns]
df_complete["HD_Preparedness"] = df_complete[prepared_cols].mean(axis=1)
df_incomplete["HD_Preparedness"] = df_incomplete[prepared_cols].mean(axis=1)

# Clean STATE (Q8) and create REGION (complete only)

df_complete["State_clean"] = (
    df_complete["Q8"]
    .astype(str)
    .str.strip()
    .str.lower()
)

state_map = {
    "ny": "NY", "nys": "NY", "new york": "NY",
    "fl": "FL", "florida": "FL",
    "ga": "GA", "georgia": "GA",
    "il": "IL", "illinois": "IL",
    "in": "IN", "indiana": "IN",
    "ma": "MA", "massachusetts": "MA",
    "ct": "CT", "connecticut": "CT",
    "wi": "WI", "wisconsin": "WI",
    "va": "VA", "virginia": "VA",
    "pa": "PA", "pennsylvania": "PA",
    "ri": "RI", "rhode island": "RI",
    "ok": "OK", "oklahoma": "OK",
    "ne": "NE", "nebraska": "NE"
}

df_complete["State"] = df_complete["State_clean"].map(state_map)

east_states = ["NY", "FL", "GA", "IL", "IN", "MA", "CT", "WI", "VA", "PA", "RI"]
west_states = ["CA", "OR", "WA", "AZ"]

df_complete["Region"] = "Other"
df_complete.loc[df_complete["State"].isin(east_states), "Region"] = "East"
df_complete.loc[df_complete["State"].isin(west_states), "Region"] = "West"

print(df_complete["Region"].value_counts())

def clean_state_region(df_subset):
    df_subset["State_clean"] = df_subset["Q8"].astype(str).str.strip().str.lower()
    df_subset["State"] = df_subset["State_clean"].map(state_map)
    df_subset["Region"] = "Other"
    df_subset.loc[df_subset["State"].isin(east_states), "Region"] = "East"
    df_subset.loc[df_subset["State"].isin(west_states), "Region"] = "West"
    return df_subset

df_complete = clean_state_region(df_complete)
df_incomplete = clean_state_region(df_incomplete)

# 6. Clinical experience, gender, grad year, unknown HD
# -----------------------------
for df_subset, name in zip([df_complete, df_incomplete], ["Complete", "Incomplete"]):
    print(f"\n{name} surveys summary:")

    # Clinical experience
    if "Clinical_Experience" in df_subset.columns:
        print("Clinical experience summary:")
        print(df_subset["Clinical_Experience"].describe())

    # Gender
    if "Gender" in df_subset.columns:
        print("\nGender counts:")
        print(df_subset["Gender"].value_counts(dropna=False))

    # Graduation year
    if "GraduationYear" in df_subset.columns:
        print("\nGraduation year summary:")
        print(df_subset["GraduationYear"].describe())

    # Count of people who did not know what HD was (HD_Familiarity = 1)
    if "HD_Familiarity" in df_subset.columns:
        unknown_hd_count = (df_subset["HD_Familiarity"] == 1).sum()
        print(f"\nNumber of people who did not know what HD was: {unknown_hd_count}")

# -----------------------------
# 7. Q53 resources (check all that apply)
# -----------------------------
if "Q53" in df.columns:
    for df_subset, name in zip([df_complete, df_incomplete], ["Complete","Incomplete"]):
        q53_all = df_subset["Q53"].dropna().astype(str)
        all_choices = []
        for entry in q53_all:
            choices = [x.strip() for x in entry.split(",")]
            all_choices.extend(choices)
        q53_counts = pd.Series(all_choices).value_counts()
        print(f"\nQ53 Resource counts ({name} surveys):")
        print(q53_counts)

# -----------------------------
# 8. Save processed data
# -----------------------------
df_complete.to_csv("data/clean_output/df_complete_scores.csv", index=False)
df_incomplete.to_csv("data/clean_output/df_incomplete_scores.csv", index=False)
print("\nProcessing complete! Composite scores, regions, clinical info, and resources included.")

# Summary for complete surveys
print(df_complete[["HD_Familiarity","Grad_Curriculum","Importance_HD_Care",
                   "HD_Exposure","HD_Preparedness"]].describe())

# Summary for incomplete surveys
print(df_incomplete[["HD_Familiarity","Grad_Curriculum","Importance_HD_Care",
                     "HD_Exposure","HD_Preparedness"]].describe())




