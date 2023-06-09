import pandas as pd
from src.common.common_util import path_to_project_root

# root directory 지정하기
root_dir = path_to_project_root('dropout_classification')

# 자료 불러오기
df_enrolled = pd.read_csv(f'{root_dir}/data/raw/merged_enrolled_std.csv')
df_graduated = pd.read_csv(f'{root_dir}/data/raw/merged_graduated_std.csv')
df_portal = pd.read_csv(f'{root_dir}/data/processed/portal_groupby.csv')
df_blackboard = pd.read_csv(f'{root_dir}/data/processed/blackboard_groupby.csv')

# 재학생 졸업생 합치기
df_std = pd.concat([df_enrolled, df_graduated], ignore_index=True)

# 포탈 로그인 기록 합치기
df_std_00 = pd.merge(df_std, df_portal,
                     how='left',
                     left_on='STD_ID',
                     right_on='NSSO002_USER_ID')

df_std_01 = pd.merge(df_std_00, df_blackboard,
                     how='left',
                     left_on='STD_ID',
                     right_on='user_id')

# csv로 저장하기
df_std_01.to_csv(f'{root_dir}/data/processed/df_dropout_classification.csv')
