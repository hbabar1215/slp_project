import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

df_data = pd.read_parquet('data/transformed_output/df_data.parquet')
df_meta = pd.read_parquet('data/transformed_output/df_meta.parquet')


df_data_select = df_data[['ResponseId', 'school_prep_score', 'huntingtons_disease_prep_score']]

### drop rows where school_prep_score or huntingtons_disease_prep_score is < 1
df_data_select = df_data_select[(df_data_select['school_prep_score'] >= 1) & (df_data_select['huntingtons_disease_prep_score'] >= 1)]

### assess correlation between school prep score and huntingtons disease prep score
correlation = df_data_select['school_prep_score'].corr(df_data_select['huntingtons_disease_prep_score'])
print(f'Correlation between school prep score and huntingtons disease prep score: {correlation}')

### scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_data_select, x='school_prep_score', y='huntingtons_disease_prep_score')
plt.title('Scatter Plot of School Prep Score vs Huntington\'s Disease Prep Score')
plt.xlabel('School Prep Score')
plt.ylabel('Huntington\'s Disease Prep Score')
plt.show()