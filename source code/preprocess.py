import pandas as pd


def clean(path='Life Expectancy Data.csv'):

    df = pd.read_csv(path)

    df.info()

    # 统计缺失值信息
    missing_info = df.isnull().sum()

    # 打印结果
    print("缺失值统计：")
    print(missing_info)

    # 去除缺失Life expectancy的列
    df = df.dropna(subset=['Life expectancy'])

    # 使用相同国家类型的众数填充缺失的GDP值
    df['GDP'] = df.groupby('Status')['GDP'].transform(lambda x: x.fillna(x.mode()))

    # 使用相同国家类型的众数填充缺失的GDP值
    df['Population'] = df.groupby('Status')['Population'].transform(lambda x: x.fillna(x.mode()))

    # 选择只包含数值列的子集，然后使用中位数填充
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 统计缺失值信息
    missing_info = df.isnull().sum()

    # 打印结果
    print("清理完成后缺失值统计：")
    print(missing_info)

    # 打印结果
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # print(df.describe())

    # return feature(df)
    return df


def feature(df):

    df['gdp_per_expense'] = df['percentage expenditure'] / df['GDP']

    df['diff_deaths'] = df['under-five deaths'] - df['infant deaths']

    df.info()

    return df


if __name__ == "__main__":
    df = clean()
    df = feature(df)
