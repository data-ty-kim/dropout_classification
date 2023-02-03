#%%
import pandas as pd

#%%
# df_covid 전처리

df_covid = pd.read_csv('covid-19.csv', low_memory=False, encoding='cp949', thousands=',')

df_covid.drop(0, inplace=True)
df_covid.columns = ['date', 'total', 'domestic', 'foreign', 'death']
df_covid.replace('-', 0, inplace=True)
df_covid.fillna(0)
df_covid['domestic'] = df_covid['domestic'].str.replace(',', '')
df_covid['domestic'] = df_covid['domestic'].fillna(0)

df_covid = df_covid.astype(
    {'date': 'datetime64', 'total': 'int32', 'foreign': 'int32', 'death': 'int32', 'domestic': 'int32'}
    )

#%%
# NARS242 전처리

df_nars = pd.read_csv('NARS242.csv',
                      dtype={'STD_ID': 'int32', 'THE_NUMBER_OF_FUNDS': 'int32',
                             'SUM_OF_FUNDS': 'int32', 'THE_NUMBER_OF_WORKS': 'int32'})

# print(len(df_nars['STD_ID']))
# print(len(df_nars['STD_ID'].unique()))    # 4242개

#%%
# REC012 전처리

df_rec = pd.read_csv('REC012.csv', low_memory=False)
df_rec = df_rec.astype(dtype={'STD_ID': 'int32', 'SEX': 'category'})

print(df_rec.isna().sum())

# print(len(df_rec['STD_ID']))
# print(len(df_rec['STD_ID'].unique()))   # 8429개

#%%
# SCH212 전처리

df_sch = pd.read_csv('SCH212.csv', dtype={1: 'int32', 2: 'int64'})

# print(df_sch.info())
# print(len(df_sch['STD_ID'].unique()))



#%%
# df_rec와 df_nars 합치기

df_merge_0 = pd.merge(df_rec, df_nars, how='left', left_on='STD_ID', right_on='STD_ID')

df_merge_0['THE_NUMBER_OF_FUNDS'] = df_merge_0['THE_NUMBER_OF_FUNDS'].fillna(0)
df_merge_0['SUM_OF_FUNDS'] = df_merge_0['SUM_OF_FUNDS'].fillna(0)
df_merge_0['THE_NUMBER_OF_WORKS'] = df_merge_0['THE_NUMBER_OF_WORKS'].fillna(0)

df_merge_0 = df_merge_0.astype({'STD_ID': 'int32', 'PROF': 'object', 'THE_NUMBER_OF_FUNDS': 'int32',
                                'SUM_OF_FUNDS': 'int32', 'THE_NUMBER_OF_WORKS': 'int32'})

# df_merge_0.info()
# print(df_merge_0[120:130])

#%%
# df_sch도 합치기

df_merge_1 = pd.merge(df_merge_0, df_sch.drop(columns='구분'), how='left',  left_on='STD_ID', right_on='STD_ID')
df_merge_1['SUM_SCH'] = df_merge_1['SUM_SCH'].fillna(0)
df_merge_1 = df_merge_1.astype({'SUM_SCH': 'int32'})

df_merge_1.info()

#%%
# dataframe 탐색하기

# print(df_merge_1.iloc[1,:])
print(df_merge_1['REC_STS_CD'].unique())