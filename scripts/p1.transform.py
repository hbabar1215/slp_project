import pandas as pd

df = pd.read_csv('data/clean_output/df.csv')

columns = df.columns.tolist()

meta_data = ['StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail', 'ExternalReference', 
             'LocationLatitude', 'LocationLongitude', 'DistributionChannel', 'UserLanguage']

data_columns = [col for col in columns if col not in meta_data]
data_columns.insert(0, 'ResponseId')

df_data = df[data_columns]
df_meta = df[meta_data]


### just keep year-month-day in RecordedDate
df_meta['RecordedDate'] = pd.to_datetime(df_meta['RecordedDate'], errors='coerce').dt.date  


### just filter for where Status is '2'
df_data = df_data[df_meta['Status'] == 2].reset_index(drop=True)


# school: 25, 26, 27, 28 
# huntingtons disease: 41, 42, 43, 47, 48


df_data['school_prep_score'] = df_data[['Q25', 'Q26_1', 'Q26_2', 'Q26_3', 'Q26_4', 'Q26_5', 'Q26_6', 'Q27_1', 'Q27_2', 'Q27_3', 'Q27_4', 'Q27_5', 'Q27_6', 'Q28']].sum(axis=1)
df_data['school_prep_score'].describe()

df_data['huntingtons_disease_prep_score'] = df_data[['Q41', 'Q42', 'Q43', 'Q47_1', 'Q47_2', 'Q47_3', 'Q47_4', 'Q47_5', 'Q48_1', 'Q48_2']].sum(axis=1)
df_data['huntingtons_disease_prep_score'].describe()



### save df_data to parquet and pkl
df_data.to_parquet('data/transformed_output/df_data.parquet', index=False)
df_data.to_pickle('data/transformed_output/df_data.pkl')
df_data.to_csv('data/transformed_output/df_data.csv', index=False)

### save df_meta to parquet and pkl
df_meta.to_parquet('data/transformed_output/df_meta.parquet', index=False)
df_meta.to_pickle('data/transformed_output/df_meta.pkl')
df_meta.to_csv('data/transformed_output/df_meta.csv', index=False)