#%%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import missingno as msno
import datetime
import math

rc('font', family="NanumGothic")
plt.style.use('fivethirtyeight')

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

# print(df_rec.isna().sum())              # 각 열별 결측치 확인
# print(len(df_rec['STD_ID']))
# print(len(df_rec['STD_ID'].unique()))   # 8429개

#%%
# SCH212 전처리

df_sch = pd.read_csv('SCH212.csv', dtype={1: 'int32', 2: 'int64'})

# print(df_sch.info())
# print(len(df_sch['STD_ID'].unique()))

#%%
# REC032-034
df_presch = pd.read_csv('REC032-034.csv', usecols=[1, 2], low_memory=False)
# print(df_presch.head())
# print(df_presch.info())     # 8429개의 ID, 8349개의 전적대 코드

df_presch['PRESCH_CD'] = df_presch['PRESCH_CD'].apply(lambda x: 1 if x == '00053120' else 0)


#%%
# REC042-044
df_chg = pd.read_csv('REC042-044.csv',
                     dtype={'YEAR': 'int32', 'TERM': 'category'})

df_chg['CHG_DT'] = df_chg['CHG_DT'].str[:10]
df_chg['CHG_DT'] = df_chg['CHG_DT'].astype('datetime64')

# for code in sorted(df_chg['REC_CHG_CD'].unique()):
#     rec_chg_nm = df_chg[df_chg['REC_CHG_CD']==code].iloc[0,5]
#     print(f'{code} - {rec_chg_nm}, ', end='')
    # 111 - 일반복학, 121 - 일반재입학, 131 - 학과/전공변경, 134 - 학위청구방식변경, 135 - 융합전공진입,
    # 136 - 융합전공포기, 141 - 지도교수1변경, 142 - 지도교수2변경, 145 - 객원연구원, 151 - 교환학생(국외),
    # 152 - 학점교류(국내), 211 - 일반휴학, 212 - 군입대휴학, 311 - 수료, 331 - 수료(복학), 341 - 수료(미등록),
    # 411 - 미등록제적, 412 - 자퇴, 413 - 휴학경과제적, 416 - 제적, 41A - 자퇴제적(사망), 601 - 석·박사통합과정진입,
    # 602 - 석·박사통합과정포기

# print(df_chg['CHG_DT'].head())
# print(df_chg[140:150])
# print(df_chg.info())  # 4774행
# print(len(df_chg['STD_ID'].unique()))   # 4754개 => 중복값 존재

# # 중복값 제거
# condition = df_chg.duplicated(subset=['STD_ID'])
# list_duplicated = list(df_chg.loc[condition]['STD_ID'])
# df_chg[df_chg['STD_ID'].isin(list_duplicated)]

df_chg.drop(
    [426, 943, 945, 947, 2007, 2010, 2014, 2089, 2272, 2557, 2831, 2859, 2997, 3239, 3693, 3758, 3826, 4560, 4732, 4734],
    axis=0, inplace=True
    )

# print(df_chg.info())    # 4754 행. 중복값 없음.

# print(df_chg.head())
diff_month = df_chg.iloc[0,7] - datetime.timedelta(days=30)
print(diff_month.strftime('%Y-%m-%d'))


#%%
# 코로나 환자 기울기 함수 만들기
def covid_slope(date_end):
    # 결측값 예외
    if pd.isnull(date_end):
        return 0
    if date_end > datetime.datetime.strptime('2023-02-02', '%Y-%m-%d'):
        return 0
    # 30일 이전 날짜와 각각의 index 추출
    if date_end < datetime.datetime.strptime('2020-02-19', '%Y-%m-%d'):
        date_start = df_covid.iloc[0, 0]
    else:
        date_start = date_end - datetime.timedelta(days=30)
    start_index = df_covid[df_covid['date'] == date_start].index[0]
    end_index = df_covid[df_covid['date'] == date_end].index[0]
    # 조회된 날짜의 dataframe 추출
    df_covid_slope = df_covid[['date', 'total']].iloc[start_index : end_index]
    # 확진자 최대최소값과 각각의 index 추출
    covid_min = df_covid_slope['total'].min()
    covid_max = df_covid_slope['total'].max()
    min_index = df_covid_slope[df_covid_slope['total'] == covid_min].index[-1]
    max_index = df_covid_slope[df_covid_slope['total'] == covid_max].index[-1]
    # 기울기 분모 0인 경우 예외 처리
    if min_index == max_index:
        return 0
    # 기울기 계산
    slope = (covid_max - covid_min) / (max_index - min_index)

    return slope

# def covid_slope_2(date_end):
#     try:
#         # 30일 이전 날짜와 각각의 index 추출
#         if date_end < datetime.datetime.strptime('2020-02-19', '%Y-%m-%d'):
#             date_start = df_covid.iloc[0, 0]
#         else:
#             date_start = date_end - datetime.timedelta(days=30)
#         start_index = df_covid[df_covid['date'] == date_start].index[0]
#         end_index = df_covid[df_covid['date'] == date_end].index[0]
#         # 조회된 날짜의 dataframe 추출
#         df_covid_slope = df_covid[['date', 'total']].iloc[start_index : end_index]
#         # 확진자 최대최소값과 각각의 index 추출
#         covid_min = df_covid_slope['total'].min()
#         covid_max = df_covid_slope['total'].max()
#         min_index = df_covid_slope[df_covid_slope['total'] == covid_min].index[-1]
#         max_index = df_covid_slope[df_covid_slope['total'] == covid_max].index[-1]
#         # 기울기 분모 0인 경우 예외 처리
#         if min_index == max_index:
#             return 0
#         # 기울기 계산
#         slope = (covid_max - covid_min) / (max_index - min_index)
#
#         return slope
#     except IndexError:
#         return 0



#%%
# df_rec와 df_nars 합치기

df_merge_0 = pd.merge(df_rec, df_nars, how='left', left_on='STD_ID', right_on='STD_ID')

df_merge_0['THE_NUMBER_OF_FUNDS'] = df_merge_0['THE_NUMBER_OF_FUNDS'].fillna(0)
df_merge_0['SUM_OF_FUNDS'] = df_merge_0['SUM_OF_FUNDS'].fillna(0)
df_merge_0['THE_NUMBER_OF_WORKS'] = df_merge_0['THE_NUMBER_OF_WORKS'].fillna(0)

df_merge_0 = df_merge_0.astype({'STD_ID': 'int32', 'PROF': 'object', 'THE_NUMBER_OF_FUNDS': 'int32',
                                'SUM_OF_FUNDS': 'int32', 'THE_NUMBER_OF_WORKS': 'int32'})

# df_merge_0.info()

#%%
# df_sch도 합치기

df_merge_1 = pd.merge(df_merge_0, df_sch.drop(columns='구분'), how='left',  left_on='STD_ID', right_on='STD_ID')
df_merge_1['SUM_SCH'] = df_merge_1['SUM_SCH'].fillna(0)
df_merge_1 = df_merge_1.astype({'SUM_SCH': 'int32'})

df_merge_1.info()

#%%
# dataframe 합치기
df_merge_2 = pd.merge(df_merge_1.drop(columns='구분'), df_presch, how='left',  left_on='STD_ID', right_on='STD_ID')
df_merge_3 = pd.merge(df_merge_2, df_chg.drop(columns='구분'), how='left',  left_on='STD_ID', right_on='STD_ID')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_merge_3.head())


#%%
df_merge_3['covid_slope'] = df_merge_3.apply(lambda row: covid_slope(row['CHG_DT']), axis=1)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df_merge_3.head())

#%%
# condition = (df_merge_3['covid_slope'] == 0) & (pd.isnull(df_merge_3['CHG_DT']) == False)
# df_merge_3[['CHG_DT', 'covid_slope']][condition]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df_merge_3[['REC_CHG_NM', 'CHG_RSN']][df_merge_3['CHG_DT'] > datetime.datetime.strptime('2023-02-02', '%Y-%m-%d')])


#%%
# dataframe 탐색하기

# 열 이름 확인
print(df_merge_3.columns)
# Index(['구분', 'STD_ID', 'REC_STS_CD', 'REC_STS_NM', 'SEX', 'DEPT_CD', 'DEG_DIV',
#        'ENT_DIV', 'PROF', 'ENT_YEAR', 'ENT_TERM', 'THE_NUMBER_OF_FUNDS',
#        'SUM_OF_FUNDS', 'THE_NUMBER_OF_WORKS', 'SUM_SCH'],
#       dtype='object')

# 학적 상태 확인
print(sorted(df_merge_3['REC_STS_CD'].unique()))
# target label:
# [101, 201, 202, 303, 304, 401, 402, 501]
# [재학, 휴학, 수료연구(휴학), 수료연구(재학), 수료, 제적, 수료연구(제적), 졸업]

# 성별 확인
print(sorted(df_merge_3['SEX'].unique()))   # [1, 2]

# 학과 코드 확인
print(len(df_merge_3['DEPT_CD'].unique()))  # 118개

# 학위 과정 확인
print(sorted(df_merge_3['DEG_DIV'].unique()))   # [10, 20, 60]

# 입학구분 확인
print(sorted(df_merge_3['ENT_DIV'].unique()))
# [101, 113, 114, 116, 117, 121, 201, 203, 204, 207, 208, 209, 901]

# 전공교수 확인
print(len(df_merge_3['PROF'].unique()))         # 1183명

# 입학연도 확인
print(sorted(df_merge_3['ENT_YEAR'].unique()))  # [2020~2022]

# 입학학기 확인
print(sorted(df_merge_3['ENT_TERM'].unique()))  # ['1R', '2R']

# 연구비 받은 횟수
print(df_merge_3['THE_NUMBER_OF_FUNDS'].min(), df_merge_3['THE_NUMBER_OF_FUNDS'].max())
# 0~200번까지

# 연구비 합
print(df_merge_3['SUM_OF_FUNDS'].min(), df_merge_3['SUM_OF_FUNDS'].max())
# 0 147905000

# 연구 횟수
print(df_merge_3['THE_NUMBER_OF_WORKS'].min(), df_merge_3['THE_NUMBER_OF_WORKS'].max())
# 0 18

# 장학금의 합
print(df_merge_3['SUM_SCH'].min(), df_merge_3['SUM_SCH'].max())
# 0 94428000

#%%
print(df_merge_3[df_merge_1['구분'] == '졸업생'].loc[:, 'REC_STS_CD'].unique())
# 501 뿐

#%%
# EDA
# target label과 feature 시각화해보기

# key: 'STD_ID'
# target label:  'REC_STS_CD' (= 'REC_STS_NM')
# features: ['구분', 'SEX', 'DEPT_CD', 'DEG_DIV',
#        'ENT_DIV', 'PROF', 'ENT_YEAR', 'ENT_TERM', 'THE_NUMBER_OF_FUNDS',
#        'SUM_OF_FUNDS', 'THE_NUMBER_OF_WORKS', 'SUM_SCH'],

#%%
# 결측치 확인
msno.matrix(df=df_merge_3.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
plt.show()

#%%
# PROF 열에서 결측치인 행만 확인하기
# with문 써서 여기서만 출력 제한 풀기
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_merge_3[df_merge_3['PROF'].isnull()].iloc[:, [1, 3, 4, 5, 6, 7, 8, 10, 11]])

#%%
f, ax = plt.subplots(1, 1, figsize=(8, 8))
df_missing = df_merge_3[df_merge_3['PROF'].isnull()]
df_missing['REC_STS_NM'].value_counts().plot.pie(
    autopct='%1.1f%%', ax=ax, shadow=True
    )
ax.set_title('Pie plot - PROF feature')
ax.set_ylabel('')

plt.show()


#%%
# EDA - target label visualization
f, ax = plt.subplots(1, 1, figsize=(8, 8))
df_merge_1['REC_STS_CD'].value_counts().plot.pie(
    autopct='%1.1f%%', ax=ax, shadow=True
    )
ax.set_title('Pie plot - Target Label')
ax.set_ylabel('')

plt.show()

# 302가 너무 적고  304도 많지 않다. 해당 부분 조금 수정이 필요하다

#%%
# 'SEX'
f, ax = plt.subplots(1, 1, figsize=(8, 8))
df_merge_1['SEX'].value_counts().plot.pie(
    autopct='%1.1f%%', ax=ax, shadow=True
)
ax.set_title('Pie plot - SEX')
ax.set_ylabel('')

plt.show()

print('남: ', len(df_merge_1[df_merge_1['SEX'] == 1]), '여: ', len(df_merge_1[df_merge_1['SEX'] == 2]))
# 성별에 따른 차이는 없을 수도 있겠단 생각이 든다. 추가적으로 1-비율 확인해볼 것. 2-전처리해서 다시 그래프보기
# 각 성별 숫자는? 성비는?

#%%
# target label 전처리하기
# [101, 201, 202, 303, 304, 401, 402, 501]
# [재학, 휴학, 수료연구(휴학), 수료연구(재학), 수료, 제적, 수료연구(제적), 졸업]
condition_1 = (df_merge_1['REC_STS_CD'] == 101) | (df_merge_1['REC_STS_CD'] == 201) | (df_merge_1['REC_STS_CD'] == 202)
condition_2 = (df_merge_1['REC_STS_CD'] == 401) | (df_merge_1['REC_STS_CD'] == 402)
condition_3 = (df_merge_1['REC_STS_CD'] == 303) | (df_merge_1['REC_STS_CD'] == 304) | (df_merge_1['REC_STS_CD'] == 501)

df_merge_1['REC_STS_CD'].loc[condition_1] = 1   # 재학&휴학
df_merge_1['REC_STS_CD'].loc[condition_2] = 2   # 제적
df_merge_1['REC_STS_CD'].loc[condition_3] = 3   # 졸업&수료
# print(df_merge_1['REC_STS_CD'].unique())

#%%
f, ax = plt.subplots(1, 2, figsize=(16, 8))
df_merge_1['REC_STS_CD'].value_counts().plot.pie(
    autopct='%1.1f%%', ax=ax[0], shadow=True
    )
ax[0].set_title('Pie plot - Target Label')
ax[0].set_ylabel('')

sns.countplot(x='SEX', hue='REC_STS_CD', data=df_merge_1, ax=ax[1])
ax[1].set_title('Sex : REC_STS_CD')

plt.show()

# 플롯 모양이 한결 낫다 그런데 제적이 5.4%밖에 안 돼서 분류가 어려울 수도 있겠다는 생각이 든다. (과적합 문제)

#%%
f, ax = plt.subplots(1, 2, figsize=(16, 8))
for i in (1, 2):
    df_pie = df_merge_1[df_merge_1['SEX'] == i]

    df_pie['REC_STS_CD'].value_counts().plot.pie(
        autopct='%1.1f%%', ax=ax[i-1], shadow=True
        )
    ax[i-1].set_title(f'Pie plot SEX == {i}')
    ax[i-1].set_ylabel('')


plt.show()

#%%
# 'DEPT_CD'
# df_heatmap= df_merge_1[['REC_STS_CD', 'DEPT_CD', 'STD_ID']].groupby(
#                 ['REC_STS_CD', 'DEPT_CD'], as_index=False).count()
#
# df_heatmap.head()
# # df_heatmap = df_heatmap.fillna(0).astype('int32')
#
# scatter plot으로 그리기
# plt.scatter(df_heatmap['DEPT_CD'], df_heatmap['REC_STS_CD'],
#             s=df_heatmap['STD_ID'], cmap='Greens', edgecolors='black', linewidth=2)
# # # plt.colorbar(label='purchase')
# plt.show()

# 각 항목별 단과대 TOP10으로 바꾸서 다시 해보기


#%%
# # 'DEG_DIV'
# df_scatter = df_merge_1[['REC_STS_CD', 'DEG_DIV', 'STD_ID']].groupby(
#                     ['REC_STS_CD', 'DEG_DIV'], as_index=False).count()
# df_scatter = df_scatter.astype({'REC_STS_CD': 'category', 'DEG_DIV': 'category', 'STD_ID': 'int32'})
# df_scatter.head()
#
#
# plt.scatter(x=df_scatter['DEG_DIV'], y=df_scatter['REC_STS_CD'], s=df_scatter['STD_ID'])
# plt.show()
# # DEG_DIV가 20일 때 제적이 조금 커보인다.

# 산점도가 적합하지 않아 아래의 막대그래프로 대체

#%%
y_position = 1.02
f, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.countplot(x='DEG_DIV', hue='REC_STS_CD', data=df_merge_1, ax=ax)
ax.set_title('DEG_DIV : REC_STS_CD', y=y_position)
ax.set_ylabel('Count')
plt.show()

#%%

f, ax = plt.subplots(1, 3, figsize=(24, 8))
j = 0
for i in (10, 20, 60):
    df_pie = df_merge_1[df_merge_1['DEG_DIV'] == i]

    df_pie['REC_STS_CD'].value_counts().plot.pie(
        autopct='%1.1f%%', ax=ax[j], shadow=True
        )
    ax[j].set_title(f'DEG_DIV == {i}')
    ax[j].set_ylabel('')

    j += 1

plt.show()


#%%
# # 'ENT_DIV'
# df_scatter = df_merge_1[['REC_STS_CD', 'ENT_DIV', 'STD_ID']].groupby(
#                     ['REC_STS_CD', 'ENT_DIV'], as_index=False).count()
# df_scatter = df_scatter.astype({'REC_STS_CD': 'category', 'ENT_DIV': 'category', 'STD_ID': 'int32'})
# # df_scatter.head()
#
#
# plt.scatter(x=df_scatter['ENT_DIV'], y=df_scatter['REC_STS_CD'], s=df_scatter['STD_ID'])
# plt.show()

# 산점도가 적합하지 않아 아래의 파이차트로 대체

#%%
# ent_div 크기만 봐서는 알 수가 없다. 비율로 확인할 것!
# [101, 113, 114, 116, 117, 121, 201, 203, 204, 207, 208, 209, 901]

# for i in [101, 113, 114, 116, 117, 121, 201, 203, 204, 207, 208, 209, 901]:
#     print(len(df_merge_1[df_merge_1['ENT_DIV']==i]))

f, ax = plt.subplots(1, 4, figsize=(32, 8))
j = 0
for i in (101, 201, 204, 901):
    df_pie = df_merge_1[df_merge_1['ENT_DIV'] == i]

    df_pie['REC_STS_CD'].value_counts().plot.pie(
        autopct='%1.1f%%', ax=ax[j], shadow=True
        )
    ax[j].set_title(f'ENT_DIV == {i}')
    ax[j].set_ylabel('')

    j += 1

plt.show()


#%%
# 'PROF'
df_prof = df_merge_1[['REC_STS_CD', 'PROF', 'STD_ID']].groupby(
                    ['REC_STS_CD', 'PROF'], as_index=False).count().sort_values(by='STD_ID', ascending=False)

# df_merge_1[['PROF', 'STD_ID']].groupby(['PROF'], as_index=False).count().sort_values(by='STD_ID', ascending=False).head()

print(df_prof[df_prof['REC_STS_CD'] == 2].describe())

# 모든 교수에게 최소 한 명은 제적, 평균은 1.47, 그런데 std가 0.95로 높지 않아서 이걸로 분류 가능할지는 의문 max는 8

#%%
# 'ENT_YEAR'
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_merge_1['ENT_YEAR'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
ax[0].set_title('The Number of Students Entered', y=y_position)
ax[0].set_ylabel('Count')

sns.countplot(x='ENT_YEAR', hue='REC_STS_CD', data=df_merge_1, ax=ax[1])
ax[1].set_title('ENT_YEAR : REC_STS_CD', y=y_position)
plt.show()


#%%
# 'ENT_TERM'
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_merge_1['ENT_TERM'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
ax[0].set_title('The Number of Students Entered', y=y_position)
ax[0].set_ylabel('Count')

sns.countplot(x='ENT_TERM', hue='REC_STS_CD', data=df_merge_1, ax=ax[1])
ax[1].set_title('ENT_TERM : REC_STS_CD', y=y_position)
plt.show()

# YEAR랑 TERM은 의미가 없을 듯?

#%%
# 'THE_NUMBER_OF_FUNDS'
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 1]['THE_NUMBER_OF_FUNDS'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 2]['THE_NUMBER_OF_FUNDS'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 3]['THE_NUMBER_OF_FUNDS'], ax=ax)
plt.legend(['재학&휴학', '제적', '졸업&수료'])
plt.show()


#%%
# 'SUM_OF_FUNDS',
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 1]['SUM_OF_FUNDS'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 2]['SUM_OF_FUNDS'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 3]['SUM_OF_FUNDS'], ax=ax)
plt.legend(['재학&휴학', '제적', '졸업&수료'])
plt.show()


#%%
# 'THE_NUMBER_OF_WORKS',
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 1]['THE_NUMBER_OF_WORKS'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 2]['THE_NUMBER_OF_WORKS'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 3]['THE_NUMBER_OF_WORKS'], ax=ax)
plt.legend(['재학&휴학', '제적', '졸업&수료'])
plt.show()

#%%
# 'SUM_SCH'],
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 1]['SUM_SCH'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 2]['SUM_SCH'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 3]['SUM_SCH'], ax=ax)
plt.legend(['재학&휴학', '제적', '졸업&수료'])
plt.show()

# 장학금의 효과는 위대했다!

#%%
# AGE
# 나이 전처리
df_merge_1['BIRTH'] = df_merge_1['BIRTH'].astype(str)
df_merge_1['AGE'] = 2022 - df_merge_1['BIRTH'].str[:4].astype('int32')

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 1]['AGE'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 2]['AGE'], ax=ax)
sns.kdeplot(df_merge_1[df_merge_1['REC_STS_CD'] == 3]['AGE'], ax=ax)
plt.legend(['재학&휴학', '제적', '졸업&수료'])
plt.show()

# #%%
# cummulative_survival_ratio = []
#
# 
# for i in range(1, 80):
#     condition = (df_merge_1['AGE'] < i)
#     cummulative_survival_ratio.append(
#         df_merge_1[df_merge_1[condition].REC_STS_CD == 2].count() / len(df_merge_1[condition]['REC_STS_CD']))
#
# plt.figure(figsize=(7, 7))
# plt.plot(cummulative_survival_ratio)
# # plt.title('Survival rate change depending on range of Age', y=1.02)
# # plt.ylabel('Survival rate')
# # plt.xlabel('Range of Age(0~x)')
# plt.show()

#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_merge_3.head())

#%%
condition_1 = (df_merge_3['REC_STS_CD'] == 101) | (df_merge_3['REC_STS_CD'] == 201) | (df_merge_3['REC_STS_CD'] == 202)
condition_2 = (df_merge_3['REC_STS_CD'] == 401) | (df_merge_3['REC_STS_CD'] == 402)
condition_3 = (df_merge_3['REC_STS_CD'] == 303) | (df_merge_3['REC_STS_CD'] == 304) | (df_merge_3['REC_STS_CD'] == 501)

df_merge_3['REC_STS_CD'].loc[condition_1] = 1   # 재학&휴학
df_merge_3['REC_STS_CD'].loc[condition_2] = 2   # 제적
df_merge_3['REC_STS_CD'].loc[condition_3] = 3   # 졸업&수료

df_merge_3['BIRTH'] = df_merge_3['BIRTH'].astype(str)
df_merge_3['AGE'] = 2022 - df_merge_3['BIRTH'].str[:4].astype('int32')
df_merge_3['AGE'] = df_merge_3['AGE'].apply(lambda x: math.trunc(x/10))

# condition_4 = df_merge_3['PROF'] > 1.47 + 3*0.95
df_merge_3['PROF'] = df_merge_3['PROF'].apply(lambda x: 1 if x>1.47 + 3*0.95 else 0)
df_merge_3['ENT_DIV'] = df_merge_3['ENT_DIV'].apply(lambda x: 500 if x in [121, 203, 207, 208, 209,113,114,116,117] else x)


#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_merge_3.head())

#%%
# One-Hot Encoding
df_merge_3 = pd.get_dummies(df_merge_3, columns=['REC_STS_CD'], prefix='REC')
df_merge_3 = pd.get_dummies(df_merge_3, columns=['SEX'], prefix='SEX')
df_merge_3 = pd.get_dummies(df_merge_3, columns=['DEG_DIV'], prefix='DEG_DIV')
df_merge_3 = pd.get_dummies(df_merge_3, columns=['ENT_DIV'], prefix='ENT_DIV')

#%%
df_final = df_merge_3.drop(
    columns=['REC_STS_NM', 'BIRTH', 'DEPT_CD', 'ENT_YEAR', 'ENT_TERM',
             'THE_NUMBER_OF_WORKS', 'YEAR', 'TERM', 'REC_CHG_CD', 'REC_CHG_NM', 'CHG_RSN', 'CHG_DT'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_final.head())

#%%
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_final[df_final['REC_STS_CD'] == 1]['covid_slope'], ax=ax)
sns.kdeplot(df_final[df_final['REC_STS_CD'] == 2]['covid_slope'], ax=ax)
sns.kdeplot(df_final[df_final['REC_STS_CD'] == 3]['covid_slope'], ax=ax)
plt.legend(['재학&휴학', '제적', '졸업&수료'])
plt.show()
