# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split, cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.common.common_util import path_to_project_root
from src.functions_dataframe import *

# %%
# root directory 지정하기
root_dir = path_to_project_root('dropout_classification')

# %%
# 데이터 불러오기
df = pd.read_csv(f'{root_dir}/data/processed/df_dropout_classification.csv',
                 dtype={'STD_ID': 'object', 'REC_STS_CD': 'category',
                        'BIRTH': 'int16', 'AGE': 'int16', 'UNIV_FROM': 'int8', 'DEPT_CD': 'object',
                        'ADPT_CD': 'object', 'SEC_REG': 'category', 'DEG_DIV': 'category',
                        'ENT_DIV': 'category', 'PROF': 'object', 'ENT_TERM': 'category',
                        'SEQ': 'int16', 'COUNT_CHG': 'int16',
                        'SCHOLARSHIP': 'int64', 'THE_NUMBER_OF_FUNDS': 'int64', 'SUM_OF_FUNDS': 'int64',
                        'THE_NUMBER_OF_WORKS': 'int16', 'GPA': 'float16', 'PORTAL_ACCESS': 'int32', 'BB_ACCESS': 'int32'
                        },
                 index_col=0
                 )
# %%
# 재학생, 졸업생 index 따로 저장하기
index_enrolled = get_index(df, '재학생')
index_graduated = get_index(df, '졸업생')

# %%
# 원-핫 인코딩
df = one_hot_at_once(df)
# 제적:1, 재학/졸업:0으로 target 칼럼 만들기
y_label = get_target(df, 'binary')
# train set 만들기
x_features = drop_col(df)

# %%
# 데이터 세트 분할 70:30
x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(x_features.loc[index_enrolled], y_label.loc[index_enrolled],
                                                            test_size=0.3, random_state=156)
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_features.loc[index_graduated], y_label.loc[index_graduated],
                                                            test_size=0.3, random_state=156)

# train, test 데이터셋 만들기
x_train = pd.concat([x_train_a, x_train_b])
x_test = pd.concat([x_test_a, x_test_b])
y_train = pd.concat([y_train_a, y_train_b])
y_test = pd.concat([y_test_a, y_test_b])

# %% 
# 스케일링 하기

# %%
# Stratified K 겹 교차 검증
kfold = StratifiedKFold(n_splits=5)
random_state = 1
classifiers = []


# %%
# XGBoost
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
classifiers.append(xgb_wrapper)

# LightGBM
lgbm_wrapper = LGBMClassifier(n_estimators=400)
classifiers.append(lgbm_wrapper)

# %%
# # SVM for Classification
# classifiers.append(SVC(random_state=random_state))
# # 의사결정나무
# classifiers.append(DecisionTreeClassifier(random_state=random_state))
# # AdaBoost
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1))
# Random Forest
classifiers.append(RandomForestClassifier(random_state=random_state))
# Extra Tree (혹은 extremely randomized trees)
classifiers.append(ExtraTreesClassifier(random_state=random_state))
# Gradient Boosting
classifiers.append(GradientBoostingClassifier(random_state=random_state))
# # Multiple Layer Perceptron (neural network)
# classifiers.append(MLPClassifier(random_state=random_state))
# # KNN
# classifiers.append(KNeighborsClassifier())
# # Logistic 회귀
# classifiers.append(LogisticRegression(random_state=random_state))
# 선형판별분석 (Linear Discriminant Analysis: LDA)
classifiers.append(LinearDiscriminantAnalysis())

# %%
# 교차 검증 수행 후 예측 성능 반환하기
cv_results = []
# for classifier in classifiers:
#     cv_results.append(cross_val_score(classifier, x_train, y=y_train, scoring="accuracy", cv=kfold, n_jobs=16))
for classifier in classifiers:
    cv_results.append(cross_validate(classifier, x_train, y=y_train,
                                     scoring=["accuracy", "precision", "recall", "f1"], cv=kfold, n_jobs=16))

# %%
# 성능의 평균과 표준편차 계산하기
# cv_means = []
# cv_std = []
# for cv_result in cv_results:
#     cv_means.append(cv_result.mean())
#     cv_std.append(cv_result.std())
cv_acc_means = []
cv_acc_std = []
cv_pre_means = []
cv_pre_std = []
cv_rec_means = []
cv_rec_std = []
cv_f1_means = []
cv_f1_std = []

for cv_result in cv_results:
    cv_acc_means.append(cv_result['test_accuracy'].mean())
    cv_acc_std.append(cv_result['test_accuracy'].std())
    cv_pre_means.append(cv_result['test_precision'].mean())
    cv_pre_std.append(cv_result['test_precision'].std())
    cv_rec_means.append(cv_result['test_recall'].mean())
    cv_rec_std.append(cv_result['test_recall'].std())
    cv_f1_means.append(cv_result['test_f1'].mean())
    cv_f1_std.append(cv_result['test_f1'].std())

# %%
# 결과를 DataFrame으로 정리하기
# cv_res = pd.DataFrame(
#     {
#         "CrossValMeans": cv_means, "CrossValerrors": cv_std,
#         "Algorithm": ["SVC", "DecisionTree", "AdaBoost", "RandomForest", "ExtraTrees", "GradientBoosting",
#                       "MultipleLayerPerceptron", "KNeighboors", "LogisticRegression", "LinearDiscriminantAnalysis"]
#     }
# )
cv_res = pd.DataFrame(
    {
        "Algorithm": ["XGBoost", "LightGBM", "RandomForest", "ExtraTrees",
                      "GradientBoosting", "LinearDiscriminantAnalysis"],
        "CV Acc Means": cv_acc_means, "CV Acc errors": cv_acc_std,
        "CV Pre Means": cv_pre_means, "CV Pre errors": cv_pre_std,
        "CV Rec Means": cv_rec_means, "CV Rec errors": cv_rec_std,
        "CV f1 Means": cv_f1_means, "CV f1 errors": cv_f1_std,
    }
)


# %%
# Accuracy 그림 그리기
fig, ax = plt.subplots(figsize=(16, 9))
# ax = sns.barplot(x="CrossValMeans", y="Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
ax = sns.barplot(x="CV Acc Means", y="Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_acc_std})
ax.set_xlabel("Mean Accuracy")
ax.set_title("Cross validation scores: Accuracy")

# 값 표기하기
for p in ax.patches:
    ax.annotate(
        format(p.get_width(), '.4f'),
        (p.get_width() - 0.03, p.get_y() + p.get_height() / 2.),
        ha='right',
        va='center'
    )

plt.show()

# %%
# Precision 그림 그리기
fig, ax = plt.subplots(figsize=(16, 9))
ax = sns.barplot(x="CV Pre Means", y="Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_pre_std})
ax.set_xlabel("Mean Precision")
ax.set_title("Cross validation scores: Precision")

# 값 표기하기
for p in ax.patches:
    ax.annotate(
        format(p.get_width(), '.4f'),
        (p.get_width() - 0.07, p.get_y() + p.get_height() / 2.),
        ha='right',
        va='center'
    )

plt.show()

# %%
# Recall 그림 그리기
fig, ax = plt.subplots(figsize=(16, 9))
ax = sns.barplot(x="CV Rec Means", y="Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_rec_std})
ax.set_xlabel("Mean Recall")
ax.set_title("Cross validation scores: Recall")

# 값 표기하기
for p in ax.patches:
    ax.annotate(
        format(p.get_width(), '.4f'),
        (p.get_width() - 0.07, p.get_y() + p.get_height() / 2.),
        ha='right',
        va='center'
    )

plt.show()

# %%
# F1 그림 그리기
fig, ax = plt.subplots(figsize=(16, 9))
ax = sns.barplot(x="CV f1 Means", y="Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_f1_std})
ax.set_xlabel("Mean F1")
ax.set_title("Cross validation scores: F1")

# 값 표기하기
for p in ax.patches:
    ax.annotate(
        format(p.get_width(), '.4f'),
        (p.get_width() - 0.05, p.get_y() + p.get_height() / 2.),
        ha='right',
        va='center'
    )

plt.show()


# %%
# ExtraTrees, RandomForest, GradientBoosting 랑 나머지 둘 중에서 성능 제일 좋은 셋 가지고 feature 중요한 거 찾아보기
# 제일 좋은 모델 셋 중에서 하이퍼파라미터 튜닝해보기