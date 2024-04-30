# %%
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.common.common_util import path_to_project_root
from src.functions_dataframe import *
import time

# %%
# root directory 지정하기
root_dir = path_to_project_root('dropout_classification')

# 데이터 불러오기
df = pd.read_csv(f'{root_dir}/data/processed/df_final.csv',
                 dtype={'STD_ID': 'object', 'REC_STS_CD': 'category',
                        'BIRTH': 'int16', 'AGE': 'int16', 'UNIV_FROM': 'int8', 'DEPT_CD': 'category',
                        'ADPT_CD': 'category', 'SEC_REG': 'category', 'DEG_DIV': 'category',
                        'ENT_DIV': 'category', 'PROF': 'object', 'ENT_TERM': 'category', 'ENT_YEAR': 'int16',
                        'SEQ': 'int16', 'COUNT_CHG': 'int16', 'loan': 'int32',
                        'SCHOLARSHIP': 'int64', 'THE_NUMBER_OF_FUNDS': 'int64', 'SUM_OF_FUNDS': 'int64',
                        'THE_NUMBER_OF_WORKS': 'int16', 'GPA': 'float16', 'PORTAL_ACCESS': 'int32', 'BB_ACCESS': 'int32'
                        },
                 index_col=0
                 )

# log 변환해주기 'PORTAL_ACCESS', 'BB_ACCESS', 'loan'
df['PORTAL_ACCESS'] = np.log1p(df['PORTAL_ACCESS'])
df['BB_ACCESS'] = np.log1p(df['BB_ACCESS'])
df['loan'] = np.log1p(df['loan'])

# 원-핫 인코딩
df = one_hot_at_once(df, dummy=True)

# %%
# 재학생, 졸업생 index 따로 저장하기
index_enrolled = get_index(df, '재학생')
index_graduated = get_index(df, '졸업생')

# 제적:1, 재학/졸업:0으로 target 칼럼 만들기
y_label = get_target(df, 'binary')
# train set 만들기
x_features = drop_col(df)

# 데이터 세트 분할 70:30
x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(x_features.loc[index_enrolled],
                                                            y_label.loc[index_enrolled],
                                                            test_size=0.3, random_state=156
                                                            )
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_features.loc[index_graduated],
                                                            y_label.loc[index_graduated],
                                                            test_size=0.3, random_state=156
                                                            )

# train, test 데이터셋 만들기
x_train = pd.concat([x_train_a, x_train_b])
x_test = pd.concat([x_test_a, x_test_b])
y_train = pd.concat([y_train_a, y_train_b])
y_test = pd.concat([y_test_a, y_test_b])

# %%
# Stratified K 겹 교차 검증
kfold = StratifiedKFold(n_splits=5)
random_state = 1
classifiers = []

# XGBoost
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
classifiers.append(xgb_wrapper)

# LightGBM
lgbm_wrapper = LGBMClassifier(n_estimators=400)
classifiers.append(lgbm_wrapper)

# AdaBoost
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                             random_state=random_state,
                             learning_rate=0.1
                             )
classifiers.append(ada_clf)

# Random Forest
rf_clf = RandomForestClassifier(random_state=random_state)
classifiers.append(rf_clf)

# Extra Tree (혹은 extremely randomized trees)
extra_clf = ExtraTreesClassifier(random_state=random_state)
classifiers.append(extra_clf)

# Gradient Boosting
gbm_clf = GradientBoostingClassifier(random_state=random_state)
classifiers.append(gbm_clf)

# %%
# 교차 검증 수행 후 예측 성능 반환하기
cv_results = {}

for classifier in classifiers:
    f1_list = cross_val_score(classifier, x_train, y_train, scoring="f1", cv=kfold, n_jobs=-1)
    cv_results[classifier.__class__.__name__] = f1_list
    print(f'\n{classifier.__class__.__name__} CV F1-score 리스트: {np.round(f1_list, 4)}')
    print(f'{classifier.__class__.__name__} CV 평균 F1-score: {np.round(np.mean(f1_list), 4)}')

# %%
# 결과를 DataFrame으로 정리하기

df_cv_result = pd.DataFrame(
    {
        "Algorithm": cv_results.keys(),
        "CV f1 Means": [np.mean(f1) for f1 in cv_results.values()],
        "CV f1 std": [np.std(f1) for f1 in cv_results.values()]
    }
)
print(df_cv_result)

# %%
# F1-score 그림 그리기
fig, ax = plt.subplots(figsize=(16, 9))
ax = sns.barplot(x="CV f1 Means", y="Algorithm", data=df_cv_result,
                 palette="Set3", orient="h", **{'xerr': df_cv_result['CV f1 std']}
                 )
ax.set_xlabel("Mean F1-score")
ax.set_title("Cross validation scores: F1-score")

# 값 표기하기
for p in ax.patches:
    ax.annotate(
        format(p.get_width(), '.4f'),
        (p.get_width() - 0.01, p.get_y() + p.get_height() / 3),
        ha='right',
        va='center'
    )

plt.show()

# %%
# 하이퍼파라미터 튜닝할 분류기

# XGBoost
xgb_wrapper = XGBClassifier(n_estimators=400, random_state=random_state, gpu_id=-1)

# LightGBM
lgbm_wrapper = LGBMClassifier(n_estimators=400, random_state=random_state)


# %%
# XGBoost Parameter Tuning
xgb_params = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 7],
    'min_child_weight': [1, 3],
    'objective': ['binary:logistic'],
    'eval_metric': ['logloss'],
    'colsample_bytree': [0.5, 0.75]
}

list_scoring = ['accuracy', 'precision', 'recall', 'f1']

start = time.time()

gridcv = GridSearchCV(xgb_wrapper, param_grid=xgb_params, cv=kfold, scoring='f1',
                      refit=True, n_jobs=-1, verbose=1)
gridcv.fit(x_train, y_train, verbose=0)

end = time.time()

print("*** Done ***")
print(f"Time elapsed {end - start: .5f} sec", '\n')
print('GridSearchCV 최적 매개변수:', gridcv.best_params_)
print('GridSearchCV 최고 scoring: {0:.4f}'.format(gridcv.best_score_))


'''
GridSearchCV 최적 매개변수: 
{
    'colsample_bytree': 0.5, 'eval_metric': 'logloss', 'learning_rate': 0.1, 
    'max_depth': 3, 'min_child_weight': 3, 'objective': 'binary:logistic'
}
GridSearchCV 최고 scoring: 0.7487
'''

# %%
# lightGBM Parameter Tuning
lgbm_params = {
    'learning_rate': [0.01, 0.2],
    'num_leaves': [32, 64],
    'max_depth': [128, 160],
    'min_child_weight': [1, 3],
    'min_child_samples': [60, 100],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.5, 1]
}

gridcv = GridSearchCV(lgbm_wrapper, param_grid=lgbm_params, cv=kfold, scoring='f1',
                      refit=True, n_jobs=-1, verbose=1)
gridcv.fit(x_train, y_train, verbose=0)

print('lightGBM 최적 매개변수:', gridcv.best_params_)
print('lightGBM 최고 f1: {0:.4f}'.format(gridcv.best_score_))

'''
lightGBM 최적 매개변수: 
{
    'colsample_bytree': 1, 'learning_rate': 0.2, 'max_depth': 128, 
    'min_child_samples': 60, 'min_child_weight': 1, 'num_leaves': 64, 'subsample': 0.8
}
lightGBM 최고 f1: 0.7428
'''

# %%
# XGBoost 최종 모델 훈련과 예측 수행

xgb_params = {
    'colsample_bytree': 0.5,
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 3,
    'objective': 'binary:logistic',
    'early_stoppings': 100
}
xgb_wrapper = XGBClassifier(n_estimators=400)
xgb_wrapper.fit(x_train, y_train, verbose=True)
w_preds = xgb_wrapper.predict(x_test)
w_pred_probs = xgb_wrapper.predict_proba(x_test)[:, 1]

# %%
# 예측 확률이 0.35보다 크면 1, 그렇지 않으면 0으로 예측값을 결정해 리스트 객체인 preds에 저장
preds = [1 if x > 0.35 else 0 for x in w_pred_probs]
print('예측값 10개만 표시:', preds[:10])

# get_clf_eval() 함수를 적용해 성능 평가
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print('오차 행렬')
    print(confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}')
    print(f'재현율: {recall:.4f}, F1: {f1:.4f}')
    print(f'AUC: {roc_auc:.4f}')


get_clf_eval(y_test, preds, w_pred_probs)
