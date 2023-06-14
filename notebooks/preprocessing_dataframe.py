# %%
import pandas as pd
from src.common.common_util import path_to_project_root

# %%
root_dir = path_to_project_root('dropout_classification')
df = pd.read_csv(f'{root_dir}/data/processed/df_dropout_classification.csv',
                 dtype={'STD_ID': 'object', 'REC_STS_CD': 'category',
                        'BIRTH': 'int16', 'AGE': 'int16', 'UNIV_FROM': 'category', 'DEPT_CD': 'object',
                        'ADPT_CD': 'object', 'SEC_REG': 'category', 'DEG_DIV': 'category',
                        'ENT_DIV': 'category', 'PROF': 'object', 'ENT_TERM': 'category',
                        'WARNING': 'int16', 'SEG': 'int16', 'COUNT_CHG': 'int16',
                        'SCHOLARSHIP': 'int64', 'THE_NUMBER_OF_FUNDS': 'int64', 'SUM_OF_FUNDS': 'int64',
                        'THE_NUMBER_OF_WORKS': 'int16', 'PORTAL_ACCESS': 'int32', 'BB_ACCESS': 'int32'
                        },
                 index_col=0
                 )

# %%
# target 학적상태코드: 학적상태명(REC_STS_CD,REC_STS_NM)
df_target_name = df[['REC_STS_CD', 'REC_STS_NM']].groupby('REC_STS_CD').max()
print(df_target_name)


# %%
# 학과코드: 학과명
# (DEPT_CD,DEPT_NM)
df_dept_names = df[['DEPT_CD', 'DEPT_NM']].groupby('DEPT_CD').max()
print(df_dept_names)

# %%
# (ADPT_CD,ADPT_NM)
df_adpt_names = df[['ADPT_CD', 'ADPT_NM']].groupby('ADPT_CD').max()
print(df_adpt_names)

# %%
# (DEG_DIV,DEG_NM)
df_deg_names = df[['DEG_DIV', 'DEG_NM']].groupby('DEG_DIV').max()
print(df_deg_names)
print(df_deg_names.info())

# %%
# (ENT_DIV,ENT_NM)
df_ent_names = df[['ENT_DIV', 'ENT_NM']].groupby('ENT_DIV').max()
print(df_ent_names)

# %%
# ENT_TERM 값을 1, 0 으로 바꿔주기
df.replace({'ENT_TERM': {'1R': 0, '2R': 1}}, inplace=True)
print(df['ENT_TERM'])
# %%
# PROF 항목은 outlier의 경우에 2 아니면 1 null은 0 그런 다음 onehot encoding
# print(df.loc[df['PROF'].isna(), 'REC_STS_NM']) # 거의 다 제적이네
df_outlier = df.loc[(df['REC_STS_CD'] == '401') | (df['REC_STS_CD'] == '402'), ['PROF', 'STD_ID']].groupby(['PROF']).count()
lowerbound = (df_outlier.describe().loc['mean'][0] + 3 * df_outlier.describe().loc['std'][0])
list_prof = df_outlier.loc[df_outlier['STD_ID'] > lowerbound].index.to_list()
# print(list_prof)

df['PROF'].fillna(0, inplace=True)
df['PROF'].astype(int)
# print(df['PROF'].isna().sum())
df['PROF_0'] = df['PROF'].apply(lambda x: 1 if x == 0 else 0)
df['PROF_1'] = df['PROF'].apply(lambda x: 1 if (x not in list_prof) & (x != 0) else 0)
df['PROF_2'] = df['PROF'].apply(lambda x: 1 if x in list_prof else 0)

# print(df.loc[df['PROF'] == 0, 'PROF_0'].head())
# print(df.loc[df['PROF'] != 0, 'PROF_0'].head())
# print(df[['PROF', 'PROF_0', 'PROF_1', 'PROF_2']].head(10))

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head(3))

# %%
# 재학생, 졸업생 index 따로 뽑기
index_enrolled = df[df['구분'] == '재학생'].index
index_graduated = df[df['구분'] == '졸업생'].index

# %%
# 원핫인코딩
# pd.get_dummies(df, columns=['ENT_YEAR'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(pd.get_dummies(df, columns=['ENT_YEAR']).head(3))


# %%
# target_label (일단 binary로 한 번 잡아봄)
df['target'] = df['REC_STS_CD'].apply(lambda x: 1 if (x == '401') | (x == '402') else 0)

# multi는 일단 보류
# df['target_multi'] = (
#     df['REC_STS_CD']
#     .apply(lambda x: 2 if (x == '401') | (x == '402') else (1 if x in ['303', '304', '501'] else 0))
# )

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df[(df['REC_STS_CD'] != '401') & (df['REC_STS_CD'] != '402')].head(3))

# %%
# drop 도 해야함함
# 한글명칭은 다 drop하기
df.drop(columns=['구분', 'STD_ID', 'REC_STS_NM', 'BIRTH', 'DEPT_NM', 'ADPT_NM', 'SEC_NM', 'DEG_NM',
                 'ENT_NM'], inplace=True)
# 원핫인코딩 했던 칼럼도 drop하기
df.drop(columns=['DEPT_CD', 'ADPT_CD', 'SEC_REG', 'DEG_DIV', 'ENT_DIV', 'ENT_YEAR'], inplace=True)
