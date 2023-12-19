import pandas as pd
import shap
from preprocess import clean
import seaborn as sns
import matplotlib.pyplot as plt
from pyecharts.charts import Map
from pyecharts import options as opts
from predict import lr, KNN, SVM, Elastic, lgbm


def kde(df):
    # 使用Seaborn绘制Life expectancy的KDE分布图
    column_to_plot = 'Life expectancy'

    sns.kdeplot(data=df[column_to_plot], fill=True, color='skyblue')

    # 添加标题和标签
    plt.title(f'KDE distribution - {column_to_plot}')
    plt.xlabel(column_to_plot)
    plt.ylabel('Density')

    # 显示图形
    plt.show()


def corr(df):

    # 计算所有特征之间的皮尔森相关系数
    correlation_matrix = df.corr()

    # 使用 Seaborn 绘制热力图
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

    # 添加标题
    plt.title('Characteristic Pearson correlation coefficient heatmap')

    # 显示图形
    plt.show()


def demonstrate(data, feature, target='Life expectancy'):
    """
    Explore the distribution of a feature and its relationship with the target.

    Parameters:
    - feature: str, the name of the feature column
    - target: str, the name of the target column
    - data: DataFrame, the input data containing the feature and target

    Returns:
    None (displays plots)
    """
    # 设置图形的布局
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # 绘制特征的分布图
    sns.histplot(data=data, x=feature, kde=True, ax=axes[0], color='blue', bins=30)
    axes[0].set_title(f'Distribution of {feature}')

    # 绘制特征与目标值的关系图
    sns.scatterplot(data=data, x=feature, y=target, color='green', ax=axes[1])
    axes[1].set_title(f'{feature} vs {target}')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()


def world_map(df):
    # 绘画人口分布图
    df['Country'].\
        replace(
        ['United States of America', 'Bolivia (Plurinational State of)', 'Venezuela (Bolivarian Republic of)',
         'Bosnia and Herzegovina', 'Brunei Darussalam', 'Central African Republic', 'Czechia', 'Equatorial Guinea',
         'Iran (Islamic Republic of)', "Lao People's Democratic Republic", 'Russian Federation',
         'United Kingdom of Great Britain and Northern Ireland',
         'Republic of Korea', 'Dominican Republic', 'South Sudan', 'United Republic of Tanzania', 'Czechia',
         'Democratic Republic of the Congo', 'Central African Republic', 'Viet Nam',
         "Democratic People's Republic of Korea",
         'Syrian Arab Republic', 'The former Yugoslav republic of Macedonia', 'Bosnia and Herzegovina',
         'Republic of Moldova', 'Denmark'],
        ['United States', 'Bolivia', 'Venezuela', 'Bosnia and Herz', 'Brunei', 'Central African Rep', 'Czech Rep',
         'Eq. Guinea', 'Iran', 'Lao PDR', 'Russia', 'United Kingdom', 'Korea', 'Dominican Rep.', 'S. Sudan', 'Tanzania',
         'Czech Rep.', 'Dem. Rep. Congo', 'Central African Rep.', 'Vietnam', "Dem. Rep. Korea", 'Syria', 'Macedonia',
         'Bosnia and Herz.', 'Moldova', 'Greenland'], inplace=True)

    Country = df.iloc[:, [0, 3]]
    Country_life = Country.groupby(['Country']).mean()
    Country_life = Country_life.reset_index()

    map = Map(init_opts=opts.InitOpts(width="1900px", height="900px", bg_color="#ADD8E6",
                                      page_title="World Map", theme="white"))
    map.add("Life Expectancy:", [list(z) for z in zip(Country_life['Country'], Country_life['Life expectancy'])],
            is_map_symbol_show=False,
            maptype="world", label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color="rgb(49,60,72)"))
    map.set_global_opts(title_opts=opts.TitleOpts(title='Global distribution'), legend_opts=opts.LegendOpts(is_show=False),
                        visualmap_opts=opts.VisualMapOpts(max_=90, min_=40))
    map.render('world_map.html')


def feature_visualization(model, X_test):
    plt.figure(figsize=(10, 6))

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_test)

    shap.plots.waterfall(shap_values[0])

    shap.plots.force(shap_values[0])

    # 获取期望值和shap值数组
    expected_value = explainer.expected_value
    shap_array = explainer.shap_values(X_test)

    # 获取前十20个对象的决策图
    shap.decision_plot(expected_value, shap_array[0:20], feature_names=list(X_test.columns))

    shap.plots.bar(shap_values)

    # Beeswarm plot
    shap.plots.beeswarm(shap_values)

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    # SHAP scatter plots
    # 结合具体feature
    # shap.plots.scatter(shap_values[:, "feature1"], ax=ax[0], show=False)
    # shap.plots.scatter(shap_values[:, "feature2"], ax=ax[1])


if __name__ == "__main__":
    df = clean()
    demonstrate(df, 'HIV/AIDS')
    demonstrate(df, 'Adult Mortality')
    demonstrate(df, 'Income composition of resources')

    # corr(df)
    # model, _, X_test = lgbm(df)
    # feature_visualization(model, X_test)
    # a = ['infant deaths', 'Hepatitis B', 'Polio']
    # kde(df)
    # world_map(df)
