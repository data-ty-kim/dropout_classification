import pandas as pd

# 자료 불러오기
df_enrolled = pd.read_csv('./data/raw/merged_enrolled_std.csv')
df_graduated = pd.read_csv('./data/raw/merged_graduated_std.csv')
df_blackboard = pd.read_csv('./data/processed/blackboard_groupby.csv')
df_portal = pd.read_csv('./data/processed/portal_groupby.csv')

# 재학생 졸업생 합치기
df_std
