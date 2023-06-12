import pandas as pd
from src.common.common_util import path_to_project_root

# root directory 지정하기
root_dir = path_to_project_root('dropout_classification')

# 자료 불러오기
df_enrolled = pd.read_csv(f'{root_dir}/data/raw/merged_enrolled_std.csv',
                          dtype={'REC_STS_CD': 'category', 'UNIV_FROM': 'category', 'DEPT_CD': 'object',
                                 'ADPT_CD': 'object', 'SEC_REG': 'category', 'DEG_DIV': 'category',
                                 'ENT_DIV': 'category', 'PROF': 'object', 'ENT_TERM': 'category'}
                          )
df_graduated = pd.read_csv(f'{root_dir}/data/raw/merged_graduated_std.csv',
                           dtype={'REC_STS_CD': 'category', 'UNIV_FROM': 'category', 'DEPT_CD': 'object',
                                  'ADPT_CD': 'object', 'SEC_REG': 'category', 'DEG_DIV': 'category',
                                  'ENT_DIV': 'category', 'PROF': 'object', 'ENT_TERM': 'category'}
                           )
df_portal = pd.read_csv(f'{root_dir}/data/processed/portal_groupby.csv')
df_blackboard = pd.read_csv(f'{root_dir}/data/processed/blackboard_groupby.csv')

# 재학생 졸업생 합치기
df_std = pd.concat([df_enrolled, df_graduated], ignore_index=True)

# 포탈 및 블랙보드 데이터프레임 칼럼명 변경
df_portal.drop(columns=['Unnamed: 0'], inplace=True)
df_portal.rename(columns={'NSSO002_USER_ID': 'STD_ID', 'NSSO920_ACCESS_DT': 'PORTAL_ACCESS'}, inplace=True)
df_blackboard.drop(columns=['Unnamed: 0'], inplace=True)
df_blackboard.rename(columns={'user_id': 'STD_ID', 'event_time': 'BB_ACCESS'}, inplace=True)

# 포탈 및 블랙보드 로그인 기록 합치기
df_std = pd.merge(df_std, df_portal,
                  how='left',
                  left_on='STD_ID',
                  right_on='STD_ID'
                  )

df_std = pd.merge(df_std, df_blackboard,
                  how='left',
                  left_on='STD_ID',
                  right_on='STD_ID'
                  )

# 포탈 및 블랙보드 로그인 기록이 없는 경우 0으로 대치하기
df_std['BB_ACCESS'].fillna(0, inplace=True)
df_std['PORTAL_ACCESS'].fillna(0, inplace=True)
df_std = df_std.astype({'BB_ACCESS': 'int32', 'PORTAL_ACCESS': 'int32'})

# csv로 저장하기
df_std.to_csv(f'{root_dir}/data/processed/df_dropout_classification.csv')
