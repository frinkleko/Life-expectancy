import pandas as pd
from preprocess import clean
from lightgbm import LGBMRegressor
from predict import lgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV


def Var_filter(df, threshold=0.001):
    # 使用方差过滤特征

    y = df.iloc[:, 3].tolist()
    X = df.drop(df.columns[3], axis=1)
    X['Country'] = LabelEncoder().fit_transform(X['Country'])
    X['Status'] = LabelEncoder().fit_transform(X['Status'])

    # 初始化VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)

    # 训练并选择特征
    X_filtered = selector.fit_transform(X)

    # 获取选择特征的索引
    selected_feature_indices = selector.get_support(indices=True)

    # 获取特征名称的列表
    feature_names = X.columns

    # 使用选择的特征索引获取特征名称
    selected_feature_names = feature_names[selected_feature_indices].tolist()

    return selected_feature_names


def RFE_(df):

    y = df.iloc[:, 3].tolist()
    X = df.drop(df.columns[3], axis=1)
    X['Country'] = LabelEncoder().fit_transform(X['Country'])
    X['Status'] = LabelEncoder().fit_transform(X['Status'])

    # 初始化LGBM模型
    lgb_model = LGBMRegressor(
        num_leaves=31,
        max_depth=6,
        learning_rate=0.01,
        n_estimators=10000,  # 使用多少个弱分类器
        objective='regression',
        boosting_type='gbdt',
        min_child_weight=2,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        bagging_fraction=0.9,
        feature_fraction=0.6,
        bagging_freq=5,
        seed=111  # 随机数种子
    )

    # 初始化RFE
    rfe = RFECV(estimator=lgb_model, min_features_to_select=10, cv=10, scoring='neg_mean_squared_error')

    # 训练RFE并选择特征
    rfe.fit(X, y)

    # 获取被选择的特征的索引
    selected_feature_indices = rfe.support_
    feature_names = X.columns
    selected_feature_names = feature_names[selected_feature_indices].tolist()

    return selected_feature_names


def importance(df):

    y = df.iloc[:, 3].tolist()
    X = df.drop(df.columns[3], axis=1)
    X['Country'] = LabelEncoder().fit_transform(X['Country'])
    X['Status'] = LabelEncoder().fit_transform(X['Status'])

    # 初始化LGBM模型
    lgb_model = LGBMRegressor(
        num_leaves=31,
        max_depth=6,
        learning_rate=0.01,
        n_estimators=10000,  # 使用多少个弱分类器
        objective='regression',
        boosting_type='gbdt',
        min_child_weight=2,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0,
        reg_lambda=1,
        bagging_fraction=0.9,
        feature_fraction=0.6,
        bagging_freq=5,
        seed=111  # 随机数种子
    )

    # 初始化SelectFromModel，使用LGBM模型
    selector = SelectFromModel(estimator=lgb_model, max_features=21)

    # 训练SelectFromModel并选择特征
    X_train_selected = selector.fit_transform(X, y)

    # 获取被选择的特征的索引
    selected_feature_indices = selector.get_support(indices=True)

    # 获取特征名称的列表
    feature_names = X.columns

    # 使用选择的特征索引获取特征名称
    selected_feature_names = feature_names[selected_feature_indices].tolist()

    return selected_feature_names


def Adversarial_Validation(df):
    pass


if __name__ == "__main__":
    df = clean()
    feature = RFE_(df)  # didn't work, no promotion
    print(len(feature))
    feature.append('Life expectancy')
    df = df[feature]
    lgbm(df)
