from src.common.common_util import path_to_project_root
import pandas as pd

# root directory 지정하기
root_dir = path_to_project_root('dropout_classification')

# 데이터 불러오기
df = pd.read_csv(f'{root_dir}/data/processed/df_dropout_classification.csv',
                 dtype={'STD_ID': 'object', 'REC_STS_CD': 'category',
                        'BIRTH': 'int16', 'AGE': 'int16', 'UNIV_FROM': 'int8', 'DEPT_CD': 'category',
                        'ADPT_CD': 'category', 'SEC_REG': 'category', 'DEG_DIV': 'category',
                        'ENT_DIV': 'category', 'PROF': 'object', 'ENT_TERM': 'category', 'ENT_YEAR': 'int16',
                        'SEQ': 'int16', 'COUNT_CHG': 'int16',
                        'SCHOLARSHIP': 'int64', 'THE_NUMBER_OF_FUNDS': 'int64', 'SUM_OF_FUNDS': 'int64',
                        'THE_NUMBER_OF_WORKS': 'int16', 'GPA': 'float16', 'PORTAL_ACCESS': 'int32', 'BB_ACCESS': 'int32'
                        },
                 index_col=0
                 )

df_lib = pd.read_csv(f'{root_dir}/data/processed/kulib_loan_list.csv',
                     dtype={'user_id': 'object', 'loan_date': 'int32'},
                     index_col=0
                     )
df_lib.rename(columns={'user_id': 'STD_ID'}, inplace=True)

df = df.merge(df_lib, how='left', on='STD_ID')
df['loan_date'].fillna(0, inplace=True)
df['loan_date'] = df['loan_date'].astype('int32')
df.rename(columns={'loan_date': 'loan'}, inplace=True)

df['SPENT'] = df.apply(lambda x: (2023 - x['ENT_YEAR']) * 12 if x['ENT_TERM'] == 0 else (2023 - x['ENT_YEAR']) * 12 - 6,
                       axis=1
                       )
df.drop(columns=['ENT_YEAR'], inplace=True)

# 선형성 있는 칼럼 drop ['THE_NUMBER_OF_FUNDS', 'SUM_OF_FUNDS', 'THE_NUMBER_OF_WORKS']
df.drop(columns=['THE_NUMBER_OF_WORKS', 'THE_NUMBER_OF_FUNDS'], inplace=True)

# csv로 저장하기
df.to_csv(f'{root_dir}/data/processed/df_final.csv')
