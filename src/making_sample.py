# This file is to make sample datasets for GitHub

import pandas as pd

df_blackboard = pd.DataFrame(
    {
        '': [0, 1, 2],
        'user_id': ['user_01', 'user_02', 'user_03'],
        'event_time': ['206', '177', '134']
     }
)

df_portal = pd.DataFrame(
    {
        '': [0, 1, 2],
        'USER_ID': ['user_01', 'user_02', 'user_03'],
        'ACCESS_DT': ['129', '360', '182']
     }
)

df_std = pd.DataFrame(
    [
        ['재학생', 'user_01', 303, 1991, 32, 0, 3999, '0217', '03', 60, 117, 112462, 2020,
         '1R', 0, 1, 0, 75065680, 66, 83703700, 8],
        ['졸업생', 'user_02', 501, 1993, 30, 0, 3999, '0217', '03', 10, 116, 112462, 2020,
         '1R', 0, 1, 0, 5077000, 91, 85844000, 8],
        ['졸업생', 'user_03', 501, 1993, 30, 0, 3999, '0217', '03', 10, 116, 112462, 2020,
         '1R', 0, 1, 0, 24426000, 77, 79272494, 8]
    ],
    columns=['', 'STD_ID', 'REC_STS_CD', 'BIRTH', 'AGE', 'UNIV_FROM', 'DEPT_CD', 'ADPT_CD', 'SEC_REG', 'DEG_DIV',
             'ENT_DIV', 'PROF', 'ENT_YEAR', 'ENT_TERM', 'WARNING', 'SEG', 'COUNT_CHG', 'SCHOLARSHIP',
             'THE_NUMBER_OF_FUNDS', 'SUM_OF_FUNDS', 'THE_NUMBER_OF_WORKS']
)

df_portal.to_csv('../data/processed/sample_portal.csv')
df_blackboard.to_csv('../data/processed/sample_blackboard.csv')
df_std.to_csv('../data/raw/sample_std.csv')