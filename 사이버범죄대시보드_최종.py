
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
        # KPI 연도 선택
        st.subheader("📊 KPI 연도 선택")
        kpi_year = st.selectbox("KPI 연도", options=years_all, index=len(years_all)-1, help="선택한 연도의 총 발생/총 검거/검거율을 표시합니다.")
        yoy_toggle = st.checkbox("전년 대비 증감(Δ) 표시", value=True)
        st.markdown("---")
        # Pie 옵션
        st.subheader("🥧 범주별 비중 설정")
        pie_year = st.selectbox("파이차트 연도 선택", options=sorted(df_yearly["연도"].unique().tolist()), index=0)
        pie_mode = st.radio("구분", options=["발생건수", "검거건수"], horizontal=True)
        top_n = st.slider("상위 N 범주 표시", min_value=5, max_value=20, value=10, step=1, help="나머지는 '기타'로 묶습니다.")
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
    # 4) KPI 박스 (선택 연도 + 전년 대비)
    # ----------------------
    def totals_for_year(df, year):
        occ = df[(df["연도"]==year) & (df["구분"]=="발생건수")]
        arr = df[(df["연도"]==year) & (df["구분"]=="검거건수")]
        if occ.empty or arr.empty:
            return np.nan, np.nan, np.nan
        total_occ = float(occ[month_order].sum(axis=1).values[0])
        total_arr = float(arr[month_order].sum(axis=1).values[0])
        rate = (total_arr/total_occ*100.0) if total_occ>0 else np.nan
        return total_occ, total_arr, rate

    # 현재 연도 KPI
    k_occ, k_arr, k_rate = totals_for_year(df_monthly, kpi_year)
    # 전년 KPI (있을 경우)
    prev_year = kpi_year - 1
    p_occ, p_arr, p_rate = (np.nan, np.nan, np.nan)
    if prev_year in years_all:
        p_occ, p_arr, p_rate = totals_for_year(df_monthly, prev_year)

    def delta_str(curr, prev, pct=False, pp=False):
        if (pd.isna(curr) or pd.isna(prev) or prev==0) and (pct or not pp):
            return "NA"
        if pp:
            diff = curr - prev if (pd.notna(curr) and pd.notna(prev)) else np.nan
            return f"{diff:+.1f}p" if pd.notna(diff) else "NA"
        diff = curr - prev if (pd.notna(curr) and pd.notna(prev)) else np.nan
        if pd.isna(diff):
            return "NA"
        if pct:
            pctv = (diff/prev*100.0) if prev!=0 else np.nan
            return f"{diff:+,.0f} ({pctv:+.1f}%)" if pd.notna(pctv) else f"{diff:+,.0f}"
        else:
            return f"{diff:+,.0f}"

    occ_delta = delta_str(k_occ, p_occ, pct=True) if yoy_toggle else None
    arr_delta = delta_str(k_arr, p_arr, pct=True) if yoy_toggle else None
    rate_delta = delta_str(k_rate, p_rate, pp=True) if yoy_toggle else None

    c1,c2,c3 = st.columns(3)
    c1.metric(f"{kpi_year}년 총 발생", f"{k_occ:,.0f}" if pd.notna(k_occ) else "NA", delta=occ_delta)
    c2.metric(f"{kpi_year}년 총 검거", f"{k_arr:,.0f}" if pd.notna(k_arr) else "NA", delta=arr_delta)
    c3.metric(f"{kpi_year}년 검거율", f"{k_rate:,.1f}%" if pd.notna(k_rate) else "NA", delta=rate_delta)

    # ----------------------
    # 5) 범주별 비중 (파이차트)
    # ----------------------
    # df_yearly는 wide 형태: [연도, 구분, <범주1> ... <범주N>]
    category_cols = [c for c in df_yearly.columns if c not in ["연도", "구분"]]
    df_sel = df_yearly[(df_yearly["연도"] == pie_year) & (df_yearly["구분"] == pie_mode)]
    if not df_sel.empty:
        s = df_sel[category_cols].sum(axis=0).sort_values(ascending=False)
        # 상위 N + 기타
        top = s.head(top_n)
        if len(s) > top_n:
            etc_val = s.iloc[top_n:].sum()
            top = pd.concat([top, pd.Series({"기타": etc_val})])
        pie_df = top.reset_index()
        pie_df.columns = ["범주", "건수"]

        fig_pie = px.pie(
            pie_df, names="범주", values="건수",
            title=f"{pie_year}년 {pie_mode} 범주별 비중",
            hole=0.35
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(l=10,r=10,t=50,b=10))

    # ----------------------
    # 레이아웃 출력
    # ----------------------
    colA, colB = st.columns([1,1])
    with colA:
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        st.plotly_chart(fig2, use_container_width=True)

    # KPI 아래에 파이차트 섹션
    st.subheader("🥧 범주별 비중")
    if 'fig_pie' in locals():
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("선택한 조건의 파이차트를 생성할 수 없습니다. 데이터 범위를 확인하세요.")

    # 히트맵
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("원본 데이터 미리보기"):
        st.write("연도별 데이터", df_yearly.head(10))
        st.write("월별 데이터", df_monthly.head(10))

else:
    st.info("좌측에서 두 개의 CSV 파일을 모두 업로드하면 대시보드가 표시됩니다.")
