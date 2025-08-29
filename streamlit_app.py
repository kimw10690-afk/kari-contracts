#######################
# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import re
from datetime import date

#######################
# Page configuration
st.set_page_config(
    page_title="KARI Contract Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded")
alt.themes.enable("default")

#######################
# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem; padding-right: 2rem;
    padding-top: 1rem;  padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] { padding-left: 0rem; padding-right: 0rem; }

/* KPI 카드 흰색 */
[data-testid="stMetric"]{
    background-color:#ffffff; text-align:center; padding:15px 0;
    border:1px solid #eaeaea; border-radius:10px; color:#111111;
}
[data-testid="stMetricLabel"]{ display:flex; justify-content:center; align-items:center; }
[data-testid="stMetricDeltaIcon-Up"],[data-testid="stMetricDeltaIcon-Down"]{
    position:relative; left:38%; transform:translateX(-50%);
}

/* 작은 정보 상자 & 기간 배지 */
.small-metric{
    background:#fff; border:1px solid #eaeaea; border-radius:10px;
    padding:10px 12px; line-height:1.25; font-size:0.9rem; overflow-wrap:anywhere;
}
.small-metric b{ font-size:0.95rem; }
.period-badge{
    display:inline-block; background:#f6f6f6; border:1px solid #eaeaea;
    border-radius:8px; padding:8px 10px; margin-bottom:10px; font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

#######################
# Helpers
def get_slider_bounds(series: pd.Series, *, round_digits: int = 1,
                      floor_zero: bool = True, step: float = 0.1):
    """안전한 슬라이더 경계 계산"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0, True
    mn = float(s.min())
    mx = float(s.max())
    if floor_zero:
        mn = max(0.0, mn)
    mn = round(mn, round_digits)
    mx = round(mx, round_digits)
    if not (mx > mn):
        mx = mn + step
    disabled = (s.nunique(dropna=True) < 2)
    return mn, mx, disabled

def fmt_total_amount_str(total_jo: float) -> str:
    """
    총액 표기: 0.1 조원 미만이면 억원으로, 아니면 조원으로.
    - 입력은 '조원' 단위 합계(total_jo)
    """
    try:
        val = float(total_jo)
    except Exception:
        return "-"
    return f"{val*1e4:,.1f} 억원" if val < 0.1 else f"{val:,.3f} 조원"

#######################
# Load data (항우연 통합 파일)
# 업로드 파일명: data2.csv
df = pd.read_csv("data2.csv", encoding="utf-8-sig")

# --- 공통 전처리 ---
# 금액
if "계약금액" in df.columns:
    df["계약금액"] = pd.to_numeric(df["계약금액"], errors="coerce")
else:
    df["계약금액"] = pd.NA

# 날짜/연도
df["_date"] = pd.to_datetime(df.get("계약일자"), errors="coerce")
df["_year"] = df["_date"].dt.year

# 단위 파생
df["계약금액_억원"] = df["계약금액"] / 1e8
df["계약금액_조원"] = df["계약금액"] / 1e12

# 계약기간(시작/종료)
df["_p_start"] = pd.to_datetime(df.get("계약시작일"), errors="coerce")
df["_p_end"]   = pd.to_datetime(df.get("계약종료일"), errors="coerce")
mask = df[["_p_start","_p_end"]].notna().all(axis=1)
df["_p_days"] = pd.NA
df.loc[mask, "_p_days"] = (df.loc[mask, "_p_end"] - df.loc[mask, "_p_start"]).dt.days + 1

# 기간 구분(단년/다년)
df["기간구분"] = "기간정보없음"
days = pd.to_numeric(df["_p_days"], errors="coerce")
ny = np.ceil(days / 365.0)
df.loc[days.notna() & (days <= 365), "기간구분"] = "단년(≤1년)"
multi_years = ny[days > 365].astype("Int64").astype(str)
df.loc[days > 365, "기간구분"] = "다년(" + multi_years + "년)"

#######################
# Sidebar
with st.sidebar:
    st.title("항우연 계약 분석 대시보드")
    st.caption("필터와 키워드를 조합해 원하는 계약만 좁혀보세요.")

    base = df.copy()

    # ------------ 필터 UI ------------
    years = sorted([int(y) for y in base["_year"].dropna().unique()]) if base["_year"].notna().any() else []
    year_sel = st.selectbox("연도 선택 (Δ는 연도 선택 시 표시)", options=["전체"] + years, index=0)

    type_options = sorted([x for x in base.get("구분", pd.Series(dtype=str)).dropna().unique()])
    type_sel = st.multiselect("구분(공사/용역/물품 등)", options=type_options, default=[])

    method_options = sorted([x for x in base.get("계약방법", pd.Series(dtype=str)).dropna().unique()])
    method_sel = st.multiselect("계약방법", options=method_options, default=[])

    # 💰 금액 슬라이더: 억원 (안전 처리)
    amt_min, amt_max, amt_disabled = get_slider_bounds(base["계약금액_억원"], round_digits=1, floor_zero=True, step=0.1)
    amt_range = st.slider(
        "계약금액 범위 (억원)",
        min_value=float(amt_min),
        max_value=float(amt_max),
        value=(float(amt_min), float(amt_max)),
        step=0.1,
        disabled=amt_disabled,
    )

    st.markdown("---")

    # ------------ 키워드 검색 ------------
    keyword = st.text_input("🔎 키워드 검색", placeholder="계약명, 거래처, 수의계약사유, 출처파일 등")
    match_mode = st.radio("일치 방식", ["AND (모두 포함)", "OR (하나라도 포함)"], horizontal=True)

    search_cols = [c for c in ["계약명","거래처","수의계약사유","구매요구부서","출처파일","시트명"] if c in base.columns]
    if search_cols:
        if "search_text" not in base.columns:
            base["search_text"] = (
                base[search_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
            )
    else:
        base["search_text"] = ""

    # ------------ 필터 적용 ------------
    filtered = base.copy()
    if year_sel != "전체":
        filtered = filtered[filtered["_year"] == int(year_sel)]
    if type_sel:
        filtered = filtered[filtered["구분"].isin(type_sel)]
    if method_sel:
        filtered = filtered[filtered["계약방법"].isin(method_sel)]
    if filtered["계약금액_억원"].notna().any():
        filtered = filtered[(filtered["계약금액_억원"] >= amt_range[0]) & (filtered["계약금액_억원"] <= amt_range[1])]

    if keyword.strip():
        tokens = [t for t in keyword.lower().split() if t]
        if tokens:
            if match_mode.startswith("AND"):
                for t in tokens:
                    filtered = filtered[filtered["search_text"].str.contains(re.escape(t), na=False)]
            else:
                pattern = "|".join(map(re.escape, tokens))
                filtered = filtered[filtered["search_text"].str.contains(pattern, na=False)]

    filtered = filtered.drop(columns=["search_text"], errors="ignore")
    st.session_state["filtered_df"] = filtered

    # Δ 계산(연도 선택 시)
    prev_year_sum_jo = None
    if year_sel != "전체":
        y = int(year_sel)
        prev_df = base.copy()
        prev_df = prev_df[prev_df["_year"] == (y - 1)]
        if type_sel:
            prev_df = prev_df[prev_df["구분"].isin(type_sel)]
        if method_sel:
            prev_df = prev_df[prev_df["계약방법"].isin(method_sel)]
        if prev_df["계약금액_억원"].notna().any():
            prev_df = prev_df[(prev_df["계약금액_억원"] >= amt_range[0]) & (prev_df["계약금액_억원"] <= amt_range[1])]
        if keyword.strip():
            if "search_text" not in prev_df.columns:
                prev_df["search_text"] = (
                    prev_df[search_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
                    if search_cols else ""
                )
            tokens = [t for t in keyword.lower().split() if t]
            if tokens:
                if match_mode.startswith("AND"):
                    for t in tokens:
                        prev_df = prev_df[prev_df["search_text"].str.contains(re.escape(t), na=False)]
                else:
                    pattern = "|".join(map(re.escape, tokens))
                    prev_df = prev_df[prev_df["search_text"].str.contains(pattern, na=False)]
            prev_df = prev_df.drop(columns=["search_text"], errors="ignore")
        prev_year_sum_jo = float(prev_df["계약금액_조원"].sum()) if "계약금액_조원" in prev_df.columns else 0.0

    # 사이드바 요약 배지 (총액 단위 자동 전환)
    if "_date" in filtered.columns and filtered["_date"].notna().any():
        start_dt = pd.to_datetime(filtered["_date"].min()).date()
        end_dt   = pd.to_datetime(filtered["_date"].max()).date()
        total_jo = float(filtered["계약금액_조원"].sum()) if filtered["계약금액_조원"].notna().any() else 0.0
        total_str = fmt_total_amount_str(total_jo)
        st.success(f"기간: {start_dt} ~ {end_dt} | 총액: {total_str} | 건수: {len(filtered):,}건")
    else:
        total_jo = float(filtered["계약금액_조원"].sum()) if "계약금액_조원" in filtered.columns else 0.0
        total_str = fmt_total_amount_str(total_jo)
        st.success(f"기간: 데이터 없음 | 총액: {total_str} | 건수: {len(filtered):,}건")

#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

# =======================
# 컬럼 0: KPI + 기간
# =======================
with col[0]:
    st.markdown("### 📌 핵심 지표")

    data = st.session_state.get("filtered_df", df).copy()

    # 기간 배지
    if "_date" in data.columns and data["_date"].notna().any():
        start_dt = pd.to_datetime(data["_date"].min()).date()
        end_dt   = pd.to_datetime(data["_date"].max()).date()
        st.markdown(f"<div class='period-badge'>기간: <b>{start_dt}</b> ~ <b>{end_dt}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='period-badge'>기간: 데이터 없음</div>", unsafe_allow_html=True)

    # KPI
    total_jo = data["계약금액_조원"].sum(skipna=True)
    avg_eok  = data["계약금액_억원"].mean(skipna=True)
    cnt      = len(data)

    delta_str = None
    if year_sel != "전체" and 'prev_year_sum_jo' in locals() and prev_year_sum_jo is not None:
        prev = prev_year_sum_jo
        if prev and prev > 0:
            delta_pct = (total_jo - prev) / prev * 100.0
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "–"

    # ✅ 총 계약 금액: 0.1조 미만이면 '억원', 아니면 '조원'
    total_display = fmt_total_amount_str(float(total_jo))
    st.metric("총 계약 금액", total_display, delta=delta_str if delta_str else None)
    st.metric("평균 계약 금액", f"{avg_eok:,.1f} 억원" if pd.notna(avg_eok) else "-")
    st.metric("계약 건수", f"{cnt:,} 건")

    if "거래처" in data.columns and not data.empty:
        top_vendor_name = (
            data.groupby("거래처")["계약금액_억원"].sum().sort_values(ascending=False).index[0]
        )
        st.markdown(f"<div class='small-metric'>최대 계약 업체(총액기준)<br><b>{top_vendor_name}</b></div>", unsafe_allow_html=True)

# =======================
# 컬럼 1: 결과 테이블 + 계약방법 Breakdown + 히트맵 + 기간통계
# =======================
with col[1]:
    st.markdown("### 🔎 검색/필터 결과 (테이블)")
    base = st.session_state.get("filtered_df", df).copy()

    # 미니필터
    with st.expander("테이블 추가 필터", expanded=False):
        q_name   = st.text_input("계약명 검색", placeholder="예: 발사관제, 연구, 유지보수 등")
        q_vendor = st.text_input("거래처(업체) 검색", placeholder="예: ㈜○○, 유한회사 ○○ 등")
        eok_min, eok_max, eok_disabled = get_slider_bounds(base["계약금액_억원"], round_digits=1, floor_zero=True, step=0.1)
        eok_range = st.slider(
            "계약금액(억원) 범위(테이블)",
            min_value=float(eok_min),
            max_value=float(eok_max),
            value=(float(eok_min), float(eok_max)),
            step=0.1,
            disabled=eok_disabled,
        )

    table_df = base.copy()
    if q_name.strip() and "계약명" in table_df.columns:
        table_df = table_df[table_df["계약명"].astype(str).str.contains(q_name, case=False, na=False)]
    if q_vendor.strip() and "거래처" in table_df.columns:
        table_df = table_df[table_df["거래처"].astype(str).str.contains(q_vendor, case=False, na=False)]
    if table_df["계약금액_억원"].notna().any():
        table_df = table_df[(table_df["계약금액_억원"] >= eok_range[0]) & (table_df["계약금액_억원"] <= eok_range[1])]

    # 정렬 + 표시 건수
    sort_dir = st.radio("금액 정렬", ["내림차순(높은→낮은)", "오름차순(낮은→높은)"],
                        horizontal=True, index=0)
    rows_opt = [50, 100, 200, 500, 1000]
    rows_to_show = st.selectbox("표시 건수(금액 기준 상위 N건)", rows_opt, index=2)

    if "계약금액_억원" in table_df.columns:
        table_df = table_df.sort_values(by="계약금액_억원",
                                        ascending=(sort_dir.startswith("오름")),
                                        na_position="last")

    show_cols = [c for c in ["계약일자","계약명","계약금액_억원","거래처","구분","계약방법","계약시작일","계약종료일","수의계약사유","출처파일","시트명"] if c in table_df.columns]
    if "계약금액_억원" in show_cols:
        table_df = table_df.rename(columns={"계약금액_억원":"계약금액(억원)"})
    if show_cols:
        show_cols = [c if c!="계약금액_억원" else "계약금액(억원)" for c in show_cols]
        table_df_display = table_df[show_cols]
        st.dataframe(
            table_df_display.head(rows_to_show),
            use_container_width=True,
            hide_index=True,
            column_config={
                "계약일자": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                "계약금액(억원)": st.column_config.NumberColumn(format="%.1f"),
            }
        )
        csv = table_df_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 현재 표 다운로드 (CSV)", data=csv, file_name="filtered_contracts.csv", mime="text/csv")
    else:
        st.info("표시할 컬럼이 없습니다.")

    st.markdown("---")

    # 📊 계약방법 Breakdown
    st.subheader("📊 계약방법 Breakdown (억원)")
    if "계약방법" in base.columns and not base.empty:
        method_summary = base.groupby("계약방법")["계약금액_억원"].sum().reset_index()
        if not method_summary.empty:
            fig_method = px.pie(
                method_summary, names="계약방법", values="계약금액_억원",
                title="계약방법별 계약금액 비율(억원)"
            )
            st.plotly_chart(fig_method, use_container_width=True)
        else:
            st.info("표시할 계약방법 데이터가 없습니다.")
    else:
        st.info("계약방법 데이터가 없습니다.")

    st.markdown("---")

    # 🎛️ 히트맵 (연도 × 계약방법)
    st.subheader("📊 연도 × 계약방법 히트맵 (억원, 라벨)")
    if "계약방법" in base.columns and "_year" in base.columns:
        pivot_df = base.groupby(["_year","계약방법"])["계약금액_억원"].sum().reset_index()
        if not pivot_df.empty:
            base_chart = alt.Chart(pivot_df).encode(
                x=alt.X("계약방법:N", title="계약 방법"),
                y=alt.Y("_year:O", title="연도"),
            )
            heat_rect = base_chart.mark_rect().encode(
                color=alt.Color("계약금액_억원:Q", title="금액(억원)", scale=alt.Scale(scheme="blues")),
                tooltip=[alt.Tooltip("_year:O", title="연도"),
                         alt.Tooltip("계약방법:N", title="방법"),
                         alt.Tooltip("계약금액_억원:Q", title="금액(억원)", format=".1f")]
            )
            heat_text = base_chart.mark_text(baseline="middle").encode(
                text=alt.Text("계약금액_억원:Q", format=".1f"),
                color=alt.value("#222")
            )
            st.altair_chart((heat_rect + heat_text).properties(width="container", height=360),
                            use_container_width=True)
        else:
            st.info("표시할 히트맵 데이터가 없습니다.")
    else:
        st.info("계약방법 / 연도 데이터가 부족하여 히트맵 표시 불가")

    st.markdown("---")

    # ⏱️ 계약기간 통계
    st.subheader("⏱️ 계약기간 통계")
    period_base = base.copy()
    if "기간구분" in period_base.columns:
        # 단년/다년 대분류
        cat = period_base["기간구분"].astype(str)
        big = np.where(cat.str.startswith("다년("), "다년", np.where(cat.eq("단년(≤1년)"), "단년", "기간정보없음"))
        period_base["기간대분류"] = big

        # 단년/다년 계약건수 표
        order = ["단년", "다년", "기간정보없음"]
        cnt_df = (period_base.groupby("기간대분류")
                  .size().reindex(order).reset_index(name="계약건수").dropna())
        st.dataframe(cnt_df, use_container_width=True, hide_index=True)

        # 최장기간 상위 N건
        top_n = st.selectbox("📜 계약기간이 긴 상위 N건", [5, 10, 20, 50], index=1)
        dur_df = period_base.copy()
        dur_df["_p_days_num"] = pd.to_numeric(dur_df["_p_days"], errors="coerce")
        long_df = (dur_df.dropna(subset=["_p_days_num"])
                   .sort_values("_p_days_num", ascending=False)
                   .head(top_n))

        if not long_df.empty:
            show_cols2 = [c for c in ["계약명","거래처"] if c in long_df.columns]
            long_df = long_df.assign(
                시작일=pd.to_datetime(long_df["_p_start"]).dt.date,
                종료일=pd.to_datetime(long_df["_p_end"]).dt.date,
                기간_일수=long_df["_p_days_num"].astype(int),
                기간_년수=(long_df["_p_days_num"]/365.0).round(1),
                계약금액_억원_round=long_df.get("계약금액_억원", pd.Series(dtype=float)).round(1)
            )
            show_cols2 += ["시작일","종료일","기간_일수","기간_년수"]
            if "계약금액_억원_round" in long_df.columns:
                long_df = long_df.rename(columns={"계약금액_억원_round":"계약금액(억원)"})
                show_cols2 += ["계약금액(억원)"]

            st.dataframe(long_df[show_cols2], use_container_width=True, hide_index=True)
        else:
            st.info("계약기간 정보를 가진 데이터가 없습니다.")
    else:
        st.info("계약기간 정보를 파싱할 수 없어 통계를 표시할 수 없습니다.")

# =======================
# 컬럼 2: 🏆 순위
# =======================
with col[2]:
    st.markdown("### 🏆 순위")
    data = st.session_state.get("filtered_df", df).copy()

    # Top 거래처(금액)
    st.subheader("🏭 Top 거래처 (금액 기준 상위 10)")
    if "거래처" in data.columns and not data.empty:
        top_vendor_amt = (data.groupby("거래처")["계약금액_억원"].sum().reset_index()
                          .sort_values("계약금액_억원", ascending=False).head(10))
        if not top_vendor_amt.empty:
            fig_vendor_amt = px.bar(top_vendor_amt, x="계약금액_억원", y="거래처", orientation="h",
                                    labels={"계약금액_억원":"계약 금액(억원)","거래처":"업체"},
                                    title="상위 10개 거래처 (금액)")
            fig_vendor_amt.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_vendor_amt, use_container_width=True)
        else:
            st.info("표시할 거래처 순위(금액) 데이터가 없습니다.")
    else:
        st.info("거래처 데이터가 없습니다.")

    # Top 거래처(건수)
    st.subheader("📦 Top 거래처 (건수 기준 상위 10)")
    if "거래처" in data.columns and not data.empty:
        top_vendor_cnt = (data.groupby("거래처").size()
                          .reset_index(name="계약건수")
                          .sort_values("계약건수", ascending=False).head(10))
        if not top_vendor_cnt.empty:
            fig_vendor_cnt = px.bar(top_vendor_cnt, x="계약건수", y="거래처", orientation="h",
                                    labels={"계약건수":"건수","거래처":"업체"},
                                    title="상위 10개 거래처 (건수)")
            fig_vendor_cnt.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_vendor_cnt, use_container_width=True)
        else:
            st.info("표시할 거래처 순위(건수) 데이터가 없습니다.")
    else:
        st.info("거래처 데이터가 없습니다.")

#######################
# 📎 부록: 원본 데이터 다운로드 (페이지 최하단)
#######################
st.markdown("---")
st.markdown("### 📎 부록: 원본 데이터 다운로드")
raw_csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("원본 CSV 다운로드", data=raw_csv, file_name="raw_data_original.csv", mime="text/csv")

st.markdown("[방사청 계약 분석 열기](https://defense-contract-analysis-dashboard.streamlit.app)")
