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


def demonstrate(df, feature):
    # 特定feature与Life expectancy的关系
    sns.histplot(df[feature], bins=20)  # 假设feature是你要查看的特征
    plt.show()

    sns.kdeplot(df[feature])
    plt.show()

    sns.boxplot(x=feature, data=df)
    plt.show()

    sns.scatterplot(x=feature, y='Life expectancy', data=df)
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


def feature_visualization(model, X_train, X_test, tree=False, kernel=False, linear=True):
    # 模型解释可视化
    if tree:
        explainer = shap.TreeExplainer(model)
    elif kernel:
        explainer = shap.KernelExplainer(model.predict, X_train)
    elif linear:
        explainer = shap.Explainer(model.predict, X_train)
    else:
        return

    # ...... need to be done


if __name__ == "__main__":
    df = clean()
    # model, X_train, X_test = lr(df)
    # feature_visualization(model, X_train, X_test)
    kde(df)
    corr(df)
    world_map(df)
