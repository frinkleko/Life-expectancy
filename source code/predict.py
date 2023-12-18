import numpy as np
from preprocess import clean
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from lightgbm import plot_importance
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


def lr(df):

    y = df['Life expectancy']

    # 创建包含剩余数据的新DataFrame
    X = df.drop('Life expectancy', axis=1)

    X['Country'] = LabelEncoder().fit_transform(X['Country'])
    X['Status'] = LabelEncoder().fit_transform(X['Status'])
    X['Year'] = OneHotEncoder().fit_transform(X['Year'].values.reshape(-1, 1)).toarray()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    Y_scaler = StandardScaler()
    y_train = Y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = Y_scaler.transform(np.array(y_test).reshape(-1, 1))

    # 使用线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 打印模型的系数和截距
    print("模型的斜率 (theta_1):", model.coef_[0])
    print("模型的截距 (theta_0):", model.intercept_)

    # 在训练集上进行预测
    y_train_pred = model.predict(X_train)
    # 在测试集上进行预测
    y_test_pred = model.predict(X_test)

    # 还原数据
    y_train = Y_scaler.inverse_transform(y_train)
    y_test = Y_scaler.inverse_transform(y_test)
    y_train_pred = Y_scaler.inverse_transform(y_train_pred)
    y_test_pred = Y_scaler.inverse_transform(y_test_pred)

    # 计算均方误差
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # 打印均方误差
    print("训练集均方误差:", mse_train)
    print("测试集均方误差:", mse_test)

    # 绘制散点图
    plt.scatter(y_test, y_test_pred, color='blue', label=f'MSE: {mse_test:.2f}', s=10)

    # 添加拟合线
    plt.plot([min(y_test), max(y_test)], [min(y_test_pred), max(y_test_pred)], linestyle='--', color='red', linewidth=2,
             label='Perfect Prediction')

    # 添加标签和标题
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('Linear regression')

    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

    return model, X_train, X_test


def Elastic(df):

    y = df['Life expectancy']

    # 创建包含剩余数据的新DataFrame
    X = df.drop('Life expectancy', axis=1)

    X['Country'] = LabelEncoder().fit_transform(X['Country'])
    X['Status'] = LabelEncoder().fit_transform(X['Status'])
    X['Year'] = OneHotEncoder().fit_transform(X['Year'].values.reshape(-1, 1)).toarray()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    Y_scaler = StandardScaler()
    y_train = Y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = Y_scaler.transform(np.array(y_test).reshape(-1, 1))

    # 4. 模型训练
    alpha_value = 0.01  # 正则化项的强度
    l1_ratio_value = 0.5  # L1 正则化的比例，范围在 0 到 1 之间
    model = ElasticNet(alpha=alpha_value, l1_ratio=l1_ratio_value)
    model.fit(X_train, y_train)

    # 在训练集上进行预测
    y_train_pred = model.predict(X_train)
    # 在测试集上进行预测
    y_test_pred = model.predict(X_test)

    # 还原数据
    y_train = Y_scaler.inverse_transform(np.array(y_train).reshape(-1, 1))
    y_test = Y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    y_train_pred = Y_scaler.inverse_transform(np.array(y_train_pred).reshape(-1, 1))
    y_test_pred = Y_scaler.inverse_transform(np.array(y_test_pred).reshape(-1, 1))

    # 计算均方误差
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # 打印均方误差
    print("训练集均方误差:", mse_train)
    print("测试集均方误差:", mse_test)

    # 绘制散点图
    plt.scatter(y_test, y_test_pred, color='blue', label=f'MSE: {mse_test:.2f}', s=10)

    # 添加拟合线
    plt.plot([min(y_test), max(y_test)], [min(y_test_pred), max(y_test_pred)], linestyle='--', color='red', linewidth=2,
             label='Perfect Prediction')

    # 添加标签和标题
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('ElasticNet')

    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

    return model, X_train, X_test


def KNN(df):
    y = df['Life expectancy']

    # 创建包含剩余数据的新DataFrame
    X = df.drop('Life expectancy', axis=1)

    X['Country'] = LabelEncoder().fit_transform(X['Country'])
    X['Status'] = LabelEncoder().fit_transform(X['Status'])
    X['Year'] = OneHotEncoder().fit_transform(X['Year'].values.reshape(-1, 1)).toarray()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    Y_scaler = MinMaxScaler()
    y_train = Y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = Y_scaler.transform(np.array(y_test).reshape(-1, 1))

    # 创建 KNN 回归模型
    k_neighbors = 10  # K 的值，即邻居的数量
    model = KNeighborsRegressor(n_neighbors=k_neighbors)
    # 训练模型
    model.fit(X_train, y_train)

    # 在训练集上进行预测
    y_train_pred = model.predict(X_train)
    # 在测试集上进行预测
    y_test_pred = model.predict(X_test)

    # 还原数据
    y_train = Y_scaler.inverse_transform(np.array(y_train).reshape(-1, 1))
    y_test = Y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    y_train_pred = Y_scaler.inverse_transform(np.array(y_train_pred).reshape(-1, 1))
    y_test_pred = Y_scaler.inverse_transform(np.array(y_test_pred).reshape(-1, 1))

    # 计算均方误差
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # 打印均方误差
    print("训练集均方误差:", mse_train)
    print("测试集均方误差:", mse_test)

    # 绘制散点图
    plt.scatter(y_test, y_test_pred, color='blue', label=f'MSE: {mse_test:.2f}', s=10)

    # 添加拟合线
    plt.plot([min(y_test), max(y_test)], [min(y_test_pred), max(y_test_pred)], linestyle='--', color='red', linewidth=2,
             label='Perfect Prediction')

    # 添加标签和标题
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('KNN')

    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

    return model, X_train, X_test


def SVM(df):

    y = df['Life expectancy']

    # 创建包含剩余数据的新DataFrame
    X = df.drop('Life expectancy', axis=1)

    X['Country'] = LabelEncoder().fit_transform(X['Country'])
    X['Status'] = LabelEncoder().fit_transform(X['Status'])
    X['Year'] = OneHotEncoder().fit_transform(X['Year'].values.reshape(-1, 1)).toarray()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    Y_scaler = StandardScaler()
    y_train = Y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = Y_scaler.transform(np.array(y_test).reshape(-1, 1))

    # 创建 NuSVR 模型
    nu_value = 0.5  # Nu 的值，范围在 0 到 1 之间
    model = NuSVR(nu=nu_value)
    model.fit(X_train, y_train)

    # 在训练集上进行预测
    y_train_pred = model.predict(X_train)
    # 在测试集上进行预测
    y_test_pred = model.predict(X_test)

    # 还原数据
    y_train = Y_scaler.inverse_transform(np.array(y_train).reshape(-1, 1))
    y_test = Y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    y_train_pred = Y_scaler.inverse_transform(np.array(y_train_pred).reshape(-1, 1))
    y_test_pred = Y_scaler.inverse_transform(np.array(y_test_pred).reshape(-1, 1))

    # 计算均方误差
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # 打印均方误差
    print("训练集均方误差:", mse_train)
    print("测试集均方误差:", mse_test)

    # 绘制散点图
    plt.scatter(y_test, y_test_pred, color='blue', label=f'MSE: {mse_test:.2f}', s=10)

    # 添加拟合线
    plt.plot([min(y_test), max(y_test)], [min(y_test_pred), max(y_test_pred)], linestyle='--', color='red', linewidth=2,
             label='Perfect Prediction')

    # 添加标签和标题
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('SVM')

    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

    return model, X_train, X_test


def lgbm(df):

    y = df['Life expectancy']

    # 创建包含剩余数据的新DataFrame
    X = df.drop('Life expectancy', axis=1)

    if 'Country' in X.columns.tolist():
        X['Country'] = LabelEncoder().fit_transform(X['Country'])
    if 'Status' in X.columns.tolist():
        X['Status'] = LabelEncoder().fit_transform(X['Status'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(
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

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100, early_stopping_rounds=200)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print('mean squared error:', mse)

    # 显示重要特征
    plot_importance(model)
    plt.show()

    # 绘制散点图
    plt.scatter(y_test, y_pred, color='blue', label=f'MSE: {mse:.2f}', s=10)

    # 添加拟合线
    plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], linestyle='--', color='red', linewidth=2,
             label='Perfect Prediction')

    # 添加标签和标题
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('LGBMRegressor')

    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

    return model, X_train, X_test


def Adversarial_Validation(df, ratio=0.2):  # lgbm version

    num_zeros = int(ratio * len(df))
    num_ones = len(df) - num_zeros
    # 生成包含 20% 0 和 80% 1 的数组
    data = np.concatenate([np.zeros(num_zeros), np.ones(num_ones)])

    # 打乱数组的顺序
    np.random.shuffle(data)

    # 将生成的数组添加为新的列
    df['New_Column'] = data

    y = df['New_Column']

    # 创建包含剩余数据的新DataFrame
    X = df.drop('Life expectancy', axis=1)

    X['Country'] = LabelEncoder().fit_transform(X['Country'])
    X['Status'] = LabelEncoder().fit_transform(X['Status'])

    if 'Country' in X.columns.tolist():
        X['Country'] = LabelEncoder().fit_transform(X['Country'])
    if 'Status' in X.columns.tolist():
        X['Status'] = LabelEncoder().fit_transform(X['Status'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)

    model = LGBMRegressor(
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

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100, early_stopping_rounds=200)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    mse = mean_squared_error(y_test, y_pred)
    print('mean squared error:', mse)

    # 显示重要特征
    plot_importance(model)
    plt.show()


if __name__ == "__main__":
    df = clean()
    # lr(df)
    # Elastic(df)
    # KNN(df)
    # SVM(df)
    lgbm(clean())  # mse:2.81
    # Adversarial_Validation(df)
