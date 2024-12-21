import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm

# 设置页面标题
st.title("加州房价预测应用")
st.write("""
    上传包含加州房屋特征的 CSV 文件，应用将使用回归模型预测房价，并生成相关图表和分析。
    另外，您还可以通过下方的输入窗口输入自己的房屋信息，预测未来十年的房价走势，并获取相关分析和解读。
""")

# 配置 Matplotlib 支持中文
def set_matplotlib_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    except Exception as e:
        st.warning("无法设置 Matplotlib 的中文字体。请确保系统中安装了 SimHei 字体。")
        st.write(f"错误详情: {e}")

set_matplotlib_font()

# 定义区域及其对应的经纬度
# 这里我们手动定义几个区域，您可以根据实际需求调整这些区域
regions = {
    "北加州（San Francisco Bay Area）": {"Latitude": 37.77, "Longitude": -122.42},
    "南加州（Los Angeles Area）": {"Latitude": 34.05, "Longitude": -118.24},
    "中加州（Central California）": {"Latitude": 36.77, "Longitude": -119.42},
    "东加州（Eastern California）": {"Latitude": 35.47, "Longitude": -118.83},
    "其他区域": {"Latitude": 36.0, "Longitude": -120.0}
}

# 加载训练好的模型
@st.cache_resource
def load_model():
    with open('california_housing_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

model = load_model()

# --------------------------
# 上传 CSV 文件部分
# --------------------------
st.header("上传您的 CSV 文件")
uploaded_file = st.file_uploader("选择一个包含房屋特征的 CSV 文件", type=["csv"])

if uploaded_file is not None:
    try:
        # 读取上传的 CSV 文件
        data = pd.read_csv(uploaded_file)
        st.subheader("上传的数据预览")
        st.write(data.head())

        # 提供“人口数”的解释
        st.markdown("""
            **人口数（Population）**：指的是特定区域（如一个街区或一个普查区）的总人口数量。这个特征反映了该区域的居住密度和人口规模，是影响房价的重要因素之一。
        """)

        # 检查必要的特征是否存在
        required_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                             'Population', 'AveOccup', 'Latitude', 'Longitude']
        missing_features = [feature for feature in required_features if feature not in data.columns]
        if missing_features:
            st.error(f"上传的文件缺少以下必要的特征: {', '.join(missing_features)}")
        else:
            # 使用模型进行预测
            predictions = model.predict(data[required_features])
            data['Predicted_MedHouseVal'] = predictions

            st.subheader("预测结果预览")
            st.write(data.head())

            # 生成图表
            st.header("数据可视化")
            # 如果实际房价存在，则进行实际值与预测值对比
            if 'MedHouseVal' in data.columns:
                st.subheader("实际房价 vs 预测房价")
                
                # Plotly 图表（支持中文）
                fig1 = px.scatter(data, x='MedHouseVal', y='Predicted_MedHouseVal',
                                  labels={'MedHouseVal': '实际房价 (万美元)', 'Predicted_MedHouseVal': '预测房价 (万美元)'},
                                  title='实际房价 vs 预测房价',
                                  hover_data=required_features)
                fig1.update_layout(
                    font=dict(
                        family="SimHei",  # 设置字体为 SimHei
                        size=12,
                        color="black"
                    )
                )
                st.plotly_chart(fig1, key='fig1')

                # 残差分布图
                data['Residuals'] = data['MedHouseVal'] - data['Predicted_MedHouseVal']
                st.subheader("残差分布")
                fig2 = px.histogram(data, x='Residuals', nbins=50, title='残差分布')
                fig2.update_layout(
                    font=dict(
                        family="SimHei",
                        size=12,
                        color="black"
                    )
                )
                st.plotly_chart(fig2, key='fig2')
            else:
                # 如果没有实际房价，展示预测房价分布
                st.subheader("预测房价分布")
                fig3 = px.histogram(data, x='Predicted_MedHouseVal', nbins=50, title='预测房价分布')
                fig3.update_layout(
                    font=dict(
                        family="SimHei",
                        size=12,
                        color="black"
                    )
                )
                st.plotly_chart(fig3, key='fig3')

            # 特征重要性
            st.subheader("特征重要性")
            if hasattr(model, 'feature_importances_'):
                feature_importances = pd.DataFrame({
                    'Feature': required_features,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                # Plotly 条形图
                fig4 = px.bar(feature_importances, x='Feature', y='Importance',
                              title='特征重要性',
                              labels={'Importance': '重要性', 'Feature': '特征'})
                fig4.update_layout(
                    font=dict(
                        family="SimHei",
                        size=12,
                        color="black"
                    )
                )
                st.plotly_chart(fig4, key='fig4')
            else:
                st.write("模型不支持获取特征重要性。")

            # 结果分析
            st.header("结果分析与解读")
            if 'MedHouseVal' in data.columns:
                mse = mean_squared_error(data['MedHouseVal'], data['Predicted_MedHouseVal'])
                r2 = r2_score(data['MedHouseVal'], data['Predicted_MedHouseVal'])
                st.write(f"**均方误差 (MSE)**: {mse:.2f}")
                st.write(f"**决定系数 (R²)**: {r2:.2f}")
                st.write("""
                    - **均方误差 (MSE)**：衡量预测值与实际值之间差异的平方平均值，数值越小表示模型性能越好。
                    - **决定系数 (R²)**：表示模型对数据的解释程度，值越接近1表示模型越好。
                """)

                # 残差分析
                st.write("""
                    **残差分析**：残差是实际值与预测值之间的差异。理想情况下，残差应当随机分布，没有明显的模式。
                """)
                fig5, ax = plt.subplots()
                sns.scatterplot(x=data['Predicted_MedHouseVal'], y=data['Residuals'], ax=ax)
                ax.axhline(0, color='red', linestyle='--')
                ax.set_xlabel('预测房价 (万美元)')
                ax.set_ylabel('残差')
                ax.set_title('预测房价 vs 残差')
                st.pyplot(fig5)  # 移除 key 参数
            else:
                st.write("由于没有实际房价数据，无法计算模型性能指标。")

    except Exception as e:
        st.error(f"发生错误: {e}")
else:
    st.info("请上传一个 CSV 文件以开始预测。")

# --------------------------
# 用户输入预测部分
# --------------------------
st.header("输入您的房屋信息进行预测")
st.write("通过下方的输入窗口，您可以输入自己的房屋信息，预测未来十年的房价走势，并获取相关分析和解读。")

# 创建用户输入表单
def user_input_form():
    st.sidebar.header("输入您的房屋特征")

    # 更改“中位数收入”为“家庭年收入”
    MedInc = st.sidebar.number_input('家庭年收入 (万美元)', min_value=0.0, max_value=15.0, value=8.3252, step=0.1)

    HouseAge = st.sidebar.slider('房屋年龄 (年)', min_value=1, max_value=100, value=41)

    AveRooms = st.sidebar.number_input('每户平均房间数', min_value=1, max_value=20, value=7, step=1)

    AveBedrms = st.sidebar.number_input('每户平均卧室数', min_value=1, max_value=10, value=1, step=1)

    Population = st.sidebar.number_input('人口数', min_value=1, max_value=10000, value=322, step=1)

    AveOccup = st.sidebar.number_input('每户平均居住人数', min_value=1, max_value=10, value=3, step=1)

    # 区域选择
    region = st.sidebar.selectbox('选择区域', list(regions.keys()))

    # 根据区域选择设置经纬度
    Latitude = regions[region]['Latitude']
    Longitude = regions[region]['Longitude']

    user_data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

input_df = user_input_form()

# 显示用户输入的特征
st.subheader('您的房屋特征')
st.write(input_df)

# 预测按钮
if st.button("预测房价"):
    try:
        # 进行预测
        prediction = model.predict(input_df)

        # 假设预测未来十年房价，可以简单地模拟增长，例如每年增长一定比例
        # 这里简单假设房价每年增长3%，十年后
        future_years = 10
        annual_growth_rate = 0.03
        future_prices = prediction[0] * ((1 + annual_growth_rate) ** np.arange(1, future_years + 1))

        # 显示预测结果
        st.subheader('预测结果')
        st.write(f"**当前预测的中位数房价**: ${prediction[0]*100000:.2f}")  # 原数据单位为10万
        st.write(f"**未来十年预测的中位数房价**: ${future_prices[-1]*100000:.2f}")

        # 生成预测趋势图
        st.subheader("未来十年房价走势预测")
        years = np.arange(1, future_years + 1)
        trend_df = pd.DataFrame({
            'Year': years,
            'Predicted_MedHouseVal': future_prices
        })

        fig_trend = px.line(trend_df, x='Year', y='Predicted_MedHouseVal',
                            labels={'Year': '年数', 'Predicted_MedHouseVal': '预测房价 (万美元)'},
                            title='未来十年房价走势预测')
        fig_trend.update_layout(
            font=dict(
                family="SimHei",
                size=12,
                color="black"
            )
        )
        st.plotly_chart(fig_trend, key='fig_trend')

        # 结果分析与解读
        st.header("结果分析与解读")
        st.write(f"""
            **当前预测的中位数房价**为 ${prediction[0]*100000:.2f}，基于您输入的房屋特征。根据预测，假设房价每年以 **3%** 的年增长率增长，未来十年的 **中位数房价** 预计将达到 ${future_prices[-1]*100000:.2f}。
        """)

        st.write("""
            ### **影响房价的主要因素**
        """)

        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.DataFrame({
                'Feature': required_features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            st.write("根据模型，以下特征对房价的影响较大：")
            st.write(feature_importances)

            # 生成特征重要性条形图
            fig_importance = px.bar(feature_importances, x='Feature', y='Importance',
                                     title='特征重要性',
                                     labels={'Importance': '重要性', 'Feature': '特征'})
            fig_importance.update_layout(
                font=dict(
                    family="SimHei",
                    size=12,
                    color="black"
                )
            )
            st.plotly_chart(fig_importance, key='fig_importance')
        else:
            st.write("模型不支持获取特征重要性。")

        st.write("""
            ### **预测结果的解读**
            
            - **家庭年收入 (MedInc)**：家庭年收入是影响房价的重要因素，收入越高，房价通常越高。
            - **房屋年龄 (HouseAge)**：较新的房屋通常价值较高，但这也取决于其他因素。
            - **每户平均房间数 (AveRooms)** 和 **每户平均卧室数 (AveBedrms)**：房间和卧室的数量也会影响房价，通常房间越多，房价越高。
            - **人口数 (Population)** 和 **每户平均居住人数 (AveOccup)**：这些因素反映了社区的密度和居住条件，也会对房价产生影响。
            - **区域 (Region)**：地理位置直接影响房价，位于优越地段的房屋通常更值钱。您选择的区域决定了房屋的经纬度，从而影响预测结果。
            
            **未来十年房价走势** 是基于假设的年增长率计算得出，实际情况可能会受到市场变化、政策调整等多种因素的影响。
        """)

    except Exception as e:
        st.error(f"预测过程中发生错误: {e}")
