import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

# -----------------------------
# 1. Load the scored data
# -----------------------------
df = pd.read_csv("data/clean_output/df_complete_scores.csv")
print("Data loaded. Shape:", df.shape)

# -----------------------------
# 2. Grad curriculum vs HD preparedness
# -----------------------------
df_corr = df[(df["Grad_Curriculum"] >= 1) & (df["HD_Preparedness"] >= 1)]
corr = df_corr["Grad_Curriculum"].corr(df_corr["HD_Preparedness"])
print("Correlation between graduate curriculum and HD preparedness:", round(corr, 3))

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_corr, x="Grad_Curriculum", y="HD_Preparedness")
plt.title("Graduate Curriculum vs HD Preparedness")
plt.xlabel("Graduate Curriculum Score")
plt.ylabel("HD Preparedness Score")
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Graduation timing vs graduate coursework (Q53 option 3)
# -----------------------------
df_q53 = df.dropna(subset=["Q9", "Q53"]).copy()
df_q53["Q9"] = pd.to_numeric(df_q53["Q9"], errors="coerce")
df_q53 = df_q53.dropna(subset=["Q9"])
median_year = df_q53["Q9"].median()
print("Graduation year range:", df_q53["Q9"].min(), "-", df_q53["Q9"].max())
print("Median graduation year:", median_year)

df_q53["Grad_Group"] = df_q53["Q9"].apply(
    lambda x: "Earlier grads" if x <= median_year else "Later grads"
)
df_q53["Graduate_Coursework"] = df_q53["Q53"].astype(str).apply(
    lambda x: "3" in [i.strip() for i in x.split(",")]
)

summary = df_q53.groupby("Grad_Group")["Graduate_Coursework"].agg(Count="sum", Total="count")
summary["Percent"] = summary["Count"] / summary["Total"] * 100
print("\nGraduate coursework selection by graduation timing:")
print(summary)

plt.figure(figsize=(6,5))
summary["Percent"].plot(kind="bar")
plt.ylabel("Percent selecting graduate coursework")
plt.title("Graduate Coursework by Graduation Timing")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

contingency = pd.crosstab(df_q53["Grad_Group"], df_q53["Graduate_Coursework"])
chi2, p, dof, expected = chi2_contingency(contingency)
print("\nChi-square test results:")
print("Chi-square statistic:", round(chi2,3))
print("p-value:", round(p,4))
print(contingency)

# -----------------------------
# 4. Graduate school settings (Q14) vs HD preparedness
# -----------------------------
df_q14 = df.dropna(subset=["Q14"]).copy()
settings = ["1","2","3","4","5"]

for s in settings:
    df_q14[f"Setting_{s}"] = df_q14["Q14"].astype(str).apply(
        lambda x: s in [i.strip() for i in x.split(",")]
    )

for s in settings:
    yes_group = df_q14.loc[df_q14[f"Setting_{s}"] == True, "HD_Preparedness"]
    no_group = df_q14.loc[df_q14[f"Setting_{s}"] == False, "HD_Preparedness"]

    plt.figure(figsize=(6,5))
    sns.boxplot(x=[0]*len(no_group) + [1]*len(yes_group),
                y=pd.concat([no_group, yes_group]))
    plt.xticks([0,1], ["No","Yes"])
    plt.title(f"HD Preparedness by Setting {s}")
    plt.xlabel("Worked in this setting during grad school?")
    plt.ylabel("HD Preparedness Score")
    plt.tight_layout()
    plt.show()

    if len(yes_group) > 1 and len(no_group) > 1:
        t_stat, p_val = ttest_ind(yes_group, no_group, equal_var=False)
        print(f"Setting {s} t-test: t={round(t_stat,3)}, p={round(p_val,4)}")

# -----------------------------
# 5. HD Exposure vs HD Preparedness
# -----------------------------
df_exp = df[(df["HD_Exposure"] >= 1) & (df["HD_Preparedness"] >= 1)].copy()
print("Rows with valid HD exposure and preparedness:", df_exp.shape[0])

# Correlation
corr_exp = df_exp["HD_Exposure"].corr(df_exp["HD_Preparedness"])
print("Correlation between HD exposure and HD preparedness:", round(corr_exp,3))

# Median split for low vs high exposure
median_exp = df_exp["HD_Exposure"].median()
df_exp["Exposure_Group"] = df_exp["HD_Exposure"].apply(
    lambda x: "Low Exposure" if x <= median_exp else "High Exposure"
)
print("Median HD exposure score:", median_exp)
print(df_exp["Exposure_Group"].value_counts())

# Boxplot by exposure group
plt.figure(figsize=(6,5))
sns.boxplot(x="Exposure_Group", y="HD_Preparedness", data=df_exp)
plt.title("HD Preparedness by HD Exposure Group")
plt.xlabel("HD Exposure Group")
plt.ylabel("HD Preparedness Score")
plt.tight_layout()
plt.show()

# T-test
low_group = df_exp[df_exp["Exposure_Group"] == "Low Exposure"]["HD_Preparedness"]
high_group = df_exp[df_exp["Exposure_Group"] == "High Exposure"]["HD_Preparedness"]
t_stat, p_val = ttest_ind(high_group, low_group, equal_var=False)
print(f"T-test results: t = {round(t_stat,3)}, p = {round(p_val,4)}")
if p_val < 0.05:
    print("Result is statistically significant: Higher exposure is associated with higher preparedness.")
else:
    print("Result is NOT statistically significant: Exposure may not be strongly linked to preparedness.")

