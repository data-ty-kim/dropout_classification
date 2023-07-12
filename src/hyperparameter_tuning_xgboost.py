from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import time
from src.common.common_util import path_to_project_root
from src.functions_dataframe import *

# root directory 지정하기
root_dir = path_to_project_root('dropout_classification')

# 데이터 불러오기
print("Load the Dataset")
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

# 재학생, 졸업생 index 따로 저장하기
index_enrolled = get_index(df, '재학생')
index_graduated = get_index(df, '졸업생')

# 원-핫 인코딩
df = one_hot_at_once(df)
# 제적:1, 재학/졸업:0으로 target 칼럼 만들기
y_label = get_target(df, 'binary')
# train set 만들기
x_features = drop_col(df)

# 데이터 세트 분할 70:30
print("Split the Dataset")
x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(x_features.loc[index_enrolled], y_label.loc[index_enrolled],
                                                            test_size=0.3, random_state=156)
x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_features.loc[index_graduated], y_label.loc[index_graduated],
                                                            test_size=0.3, random_state=156)

# train, test 데이터셋 만들기
x_train = pd.concat([x_train_a, x_train_b])
x_test = pd.concat([x_test_a, x_test_b])
y_train = pd.concat([y_train_a, y_train_b])
y_test = pd.concat([y_test_a, y_test_b])

# Stratified K 겹 교차 검증
kfold = StratifiedKFold(n_splits=5)
random_state = 1

# 하이퍼파라미터 튜닝할 분류기 - XGBoost
xgb_wrapper = XGBClassifier(n_estimators=400, random_state=random_state, early_stopping_rounds=100)

# XGBoost Parameter Tuning
xgb_params = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': range(3, 8, 1),
    'objective': ['binary:logistic', 'binary:logitraw', 'binary:hinge'],
    'eval_metric': ['logloss', 'error'],
    'min_child_weight': [1, 2, 3],
    'colsample_bytree': [0.25, 0.5, 0.75, 1]
}

# Accuracy
print("Hyperparameter tuning for Accuracy")
start = time.time()

gridcv = GridSearchCV(xgb_wrapper, param_grid=xgb_params, cv=kfold, scoring="accuracy", n_jobs=16, verbose=0)
gridcv.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])

df_gridcv_gb = pd.DataFrame(
    data=gridcv.best_params_,
    columns=['score', 'learning_rate', 'max_depth', 'objective', 'eval_metric', 'min_child_weight', 'colsample_bytree'],
    index=['accuracy']
)
df_gridcv_gb.loc['accuracy', 'score'] = gridcv.best_score_

end = time.time()
print("*** Done ***")
print(f"Time elapsed {end - start: .5f} sec", '\n')

# Precision
print("Hyperparameter tuning for Precision")
start = time.time()

gridcv = GridSearchCV(xgb_wrapper, param_grid=xgb_params, cv=kfold, scoring="precision", n_jobs=16, verbose=0)
gridcv.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])

df_gridcv_gb.loc['precision'] = gridcv.best_params_
df_gridcv_gb.loc['precision', 'score'] = gridcv.best_score_

end = time.time()
print("*** Done ***")
print(f"Time elapsed {end - start: .5f} sec", '\n')

# Recall
print("Hyperparameter tuning for Recall")
start = time.time()

gridcv = GridSearchCV(xgb_wrapper, param_grid=xgb_params, cv=kfold, scoring="recall", n_jobs=16, verbose=0)
gridcv.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])

df_gridcv_gb.loc['recall'] = gridcv.best_params_
df_gridcv_gb.loc['recall', 'score'] = gridcv.best_score_

end = time.time()
print("*** Done ***")
print(f"Time elapsed {end - start: .5f} sec", '\n')

# F1
print("Hyperparameter tuning for F1")
start = time.time()

gridcv = GridSearchCV(xgb_wrapper, param_grid=xgb_params, cv=kfold, scoring="f1", n_jobs=16, verbose=0)
gridcv.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])

df_gridcv_gb.loc['f1'] = gridcv.best_params_
df_gridcv_gb.loc['f1', 'score'] = gridcv.best_score_

end = time.time()
print("*** Done ***")
print(f"Time elapsed {end - start: .5f} sec", '\n')

# 결괏값 출력하기
print('XGBoost 최적 매개변수와 점수:')
print(df_gridcv_gb)
