
# streamlit_app.py
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="사이버 범죄 대시보드", page_icon="🚓", layout="wide")
st.title("🚓 사이버 범죄 발생·검거 현황 대시보드")

with st.sidebar:
    st.header("📥 데이터 업로드")
    yearly_file = st.file_uploader("연도별 사이버 범죄 통계 CSV", type=["csv"], key="yearly")
    monthly_file = st.file_uploader("월별 사이버 범죄 발생·검거 CSV", type=["csv"], key="monthly")
    st.caption("• 인코딩 문제 시 CP949, UTF-8(BOM)을 자동 시도합니다.\n• 열 이름 예시: 연도, 구분, 1월~12월")

def read_csv_auto(buf):
    if buf is None:
        return None
    for enc in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(buf, encoding=enc)
        except Exception:
            buf.seek(0)
    raise ValueError("CSV 인코딩을 판별할 수 없습니다. CP949 또는 UTF-8로 저장 후 다시 업로드 해주세요.")

df_yearly = read_csv_auto(yearly_file) if yearly_file else None
df_monthly = read_csv_auto(monthly_file) if monthly_file else None

def col_is_month(c):
    return str(c).endswith("월") and str(c).replace("월","").isdecimal()

if df_yearly is not None and df_monthly is not None:
    # ----------------------
    # 0) 기본 전처리
    # ----------------------
    # 월 컬럼 정렬
    month_cols = [c for c in df_monthly.columns if col_is_month(c)]
    month_order = sorted(month_cols, key=lambda x: int(str(x).replace("월","")))
    # 연도 범위
    years_all = sorted(df_monthly["연도"].unique().tolist())
    y_min, y_max = min(years_all), max(years_all)

    with st.sidebar:
        st.header("⚙️ 보기 옵션")
        year_range = st.slider("히트맵 표시 연도 범위", min_value=int(y_min), max_value=int(y_max), value=(int(y_min), int(y_max)), step=1)
        show_labels = st.checkbox("히트맵 값 표시", value=True)
        st.markdown("---")
        st.caption("제작: Plotly 기반 (폰트 설치 불필요)")

    # ----------------------
    # 1) 연도별 총 발생/검거 추세 (라인)
    # ----------------------
    yearly_summary = df_yearly.groupby(["연도", "구분"]).sum(numeric_only=True).reset_index()
    yearly_totals = yearly_summary.melt(id_vars=["연도", "구분"], var_name="범죄유형", value_name="건수")
    yearly_trend = yearly_totals.groupby(["연도", "구분"], as_index=False)["건수"].sum()

    fig1 = px.line(
        yearly_trend.sort_values("연도"),
        x="연도", y="건수", color="구분", markers=True,
        title="연도별 총 발생/검거 추세"
    )
    fig1.update_layout(margin=dict(l=10,r=10,t=50,b=10), legend_title_text="구분")

    # ----------------------
    # 2) 월별 평균 발생 패턴 (라인)
    # ----------------------
    monthly_occ = df_monthly[df_monthly["구분"]=="발생건수"].copy()
    monthly_mean = monthly_occ[month_order].mean(axis=0).reset_index()
    monthly_mean.columns = ["월", "평균 발생건수"]
    # 월 정렬용 숫자 추가
    monthly_mean["월순서"] = monthly_mean["월"].str.replace("월","").astype(int)
    monthly_mean = monthly_mean.sort_values("월순서")

    fig2 = px.line(
        monthly_mean, x="월", y="평균 발생건수", markers=True,
        title="월별 평균 발생 패턴"
    )
    fig2.update_layout(margin=dict(l=10,r=10,t=50,b=10))

    # ----------------------
    # 3) 연도·월별 발생 대비 검거율 히트맵
    # ----------------------
    # 연도별로 발생/검거 매트릭스 추출
    def build_rate_df(df):
        years = sorted(df["연도"].unique().tolist())
        rows = []
        for y in years:
            occ_row = df[(df["연도"]==y) & (df["구분"]=="발생건수")][month_order]
            arr_row = df[(df["연도"]==y) & (df["구분"]=="검거건수")][month_order]
            if occ_row.empty or arr_row.empty:
                continue
            occur = occ_row.values.flatten().astype(float)
            arrest = arr_row.values.flatten().astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                rate = np.where(occur>0, (arrest/occur)*100.0, np.nan)
            rows.append(pd.Series(rate, index=month_order, name=y))
        rate_df = pd.DataFrame(rows)
        return rate_df

    rate_df = build_rate_df(df_monthly)
    # 연도 필터
    rate_df = rate_df.loc[(rate_df.index>=year_range[0]) & (rate_df.index<=year_range[1])]

    fig3 = px.imshow(
        rate_df.values,
        labels=dict(x="월", y="연도", color="검거율(%)"),
        x=month_order,
        y=rate_df.index.astype(int),
        color_continuous_scale="YlGnBu",
        aspect="auto",
        text_auto=".1f" if show_labels else False
    )
    fig3.update_layout(title="연도·월별 발생 대비 검거율 히트맵", margin=dict(l=10,r=10,t=50,b=10))

    # ----------------------
    # 4) KPI 박스 (선택)
    # ----------------------
    latest_year = int(df_monthly["연도"].max())
    ly_occ = df_monthly[(df_monthly["연도"]==latest_year) & (df_monthly["구분"]=="발생건수")][month_order].sum(axis=1).values
    ly_arr = df_monthly[(df_monthly["연도"]==latest_year) & (df_monthly["구분"]=="검거건수")][month_order].sum(axis=1).values
    total_occ = int(ly_occ[0]) if len(ly_occ) else np.nan
    total_arr = int(ly_arr[0]) if len(ly_arr) else np.nan
    rate_latest = (total_arr/total_occ*100.0) if (total_occ and total_occ>0) else np.nan

    c1,c2,c3 = st.columns(3)
    c1.metric(f"{latest_year}년 총 발생", f"{total_occ:,.0f}" if pd.notna(total_occ) else "NA")
    c2.metric(f"{latest_year}년 총 검거", f"{total_arr:,.0f}" if pd.notna(total_arr) else "NA")
    c3.metric(f"{latest_year}년 검거율", f"{rate_latest:,.1f}%" if pd.notna(rate_latest) else "NA")

    # ----------------------
    # 레이아웃 출력
    # ----------------------
    colA, colB = st.columns([1,1])
    with colA:
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        st.plotly_chart(fig2, use_container_width=True)

    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("원본 데이터 미리보기"):
        st.write("연도별 데이터", df_yearly.head(10))
        st.write("월별 데이터", df_monthly.head(10))

else:
    st.info("좌측에서 두 개의 CSV 파일을 모두 업로드하면 대시보드가 표시됩니다.")
