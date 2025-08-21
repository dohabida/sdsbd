# ===============================
# 설치 코드 (터미널/코랩에서 실행)
# ===============================
# pip install streamlit pandas matplotlib seaborn koreanize-matplotlib

# ===============================
# streamlit_app.py
# ===============================
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

# -------------------------------
# 페이지 설정
# -------------------------------
st.set_page_config(page_title="사이버 범죄 대시보드", layout="wide")
st.title("🚓 사이버 범죄 발생·검거 현황 대시보드")

# -------------------------------
# 파일 업로드
# -------------------------------
yearly_file = st.file_uploader("연도별 사이버 범죄 통계 CSV 업로드", type=["csv"], key="yearly")
monthly_file = st.file_uploader("월별 사이버 범죄 발생·검거 CSV 업로드", type=["csv"], key="monthly")

if yearly_file and monthly_file:
    df_yearly = pd.read_csv(yearly_file, encoding="cp949")
    df_monthly = pd.read_csv(monthly_file, encoding="cp949")

    # =============================
    # 1. 연도별 총 발생/검거 추세
    # =============================
    yearly_summary = df_yearly.groupby(["연도", "구분"]).sum(numeric_only=True).reset_index()
    yearly_totals = yearly_summary.melt(id_vars=["연도", "구분"], var_name="범죄유형", value_name="건수")
    yearly_trend = yearly_totals.groupby(["연도", "구분"])["건수"].sum().reset_index()

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=yearly_trend, x="연도", y="건수", hue="구분", marker="o", ax=ax1)
    ax1.set_title("연도별 총 발생/검거 추세")
    st.pyplot(fig1)

    # =============================
    # 2. 월별 평균 발생 패턴
    # =============================
    monthly_summary = df_monthly[df_monthly["구분"]=="발생건수"].set_index("연도")
    monthly_data = monthly_summary.iloc[:,1:13]  # 1월~12월
    monthly_mean = monthly_data.mean().reset_index()
    monthly_mean.columns = ["월", "평균 발생건수"]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=monthly_mean, x="월", y="평균 발생건수", marker="o", color="red", ax=ax2)
    ax2.set_title("월별 평균 발생 패턴")
    st.pyplot(fig2)

    # =============================
    # 3. 발생 대비 검거율 히트맵
    # =============================
    heatmap_data = df_monthly.copy()
    years = heatmap_data["연도"].unique()

    rate_matrix = []
    for year in years:
        occur = heatmap_data[(heatmap_data["연도"]==year) & (heatmap_data["구분"]=="발생건수")].iloc[:,2:14].values.flatten()
        arrest = heatmap_data[(heatmap_data["연도"]==year) & (heatmap_data["구분"]=="검거건수")].iloc[:,2:14].values.flatten()
        rate = (arrest / occur * 100).round(1)
        rate_matrix.append(rate)

    rate_df = pd.DataFrame(rate_matrix, index=years, columns=[f"{i}월" for i in range(1,13)])

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(rate_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax3)
    ax3.set_title("연도별·월별 발생 대비 검거율 (%)")
    st.pyplot(fig3)

else:
    st.info("좌측 사이드바에서 두 개의 CSV 파일을 모두 업로드하세요.")
