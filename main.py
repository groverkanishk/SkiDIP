import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

st.set_page_config(
    page_title="SkiDIP",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEW_RAW_PATH = "data/new data.csv"
NEW_CLEAN_PATH = "data/new cleaned_data.csv"
PAST_RAW_PATH = "data/past data.csv"
PAST_CLEAN_PATH = "data/past cleaned data.csv"

def extract_data(file_path):
    return pd.read_csv(file_path)

def transform_past_data(data):
    data = data.copy()
    data.columns = data.columns.str.upper().str.strip()
    data.dropna(how="all", inplace=True)
    data.drop_duplicates(inplace=True)

    mean_columns = [
        "AUTOMATION_RISK_PERCENT", "AI_REPLACEMENT_SCORE",
        "SKILL_GAP_INDEX", "SALARY_BEFORE_USD", "SALARY_AFTER_USD",
        "SALARY_CHANGE_PERCENT", "SKILL_DEMAND_GROWTH_PERCENT",
        "REMOTE_FEASIBILITY_SCORE", "AI_ADOPTION_LEVEL",
        "EDUCATION_REQUIREMENT_LEVEL",
    ]
    for col in mean_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    valid_cols = [c for c in mean_columns if c in data.columns]
    industry_means = data.groupby("INDUSTRY")[valid_cols].mean()

    result_data = []
    for industry in data["INDUSTRY"].unique():
        industry_data = data[data["INDUSTRY"] == industry]
        result_data.append(industry_data)

        mean_row = {"JOB_ROLE": f"MEAN ({industry})", "INDUSTRY": industry}
        for col in data.columns:
            if col in valid_cols and industry in industry_means.index:
                mean_row[col] = round(industry_means.loc[industry, col], 2)
            elif col not in ["JOB_ROLE", "INDUSTRY"]:
                mean_row[col] = ""
        result_data.append(pd.DataFrame([mean_row]))

    return pd.concat(result_data, ignore_index=True)

def transform_new_data(data):
    data = data.copy()
    data.columns = data.columns.str.upper().str.strip()
    data.dropna(how="all", inplace=True)
    data.drop_duplicates(inplace=True)

    mean_columns = [
        "MEDIAN SALARY (USD)",
        "EXPERIENCE REQUIRED (YEARS)",
        "JOB OPENINGS (2024)",
        "PROJECTED OPENINGS (2030)",
        "REMOTE WORK RATIO (%)",
        "AUTOMATION RISK (%)",
        "GENDER DIVERSITY (%)",
        "OPENING GROWTH (%)",
    ]
    for col in mean_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    valid_cols = [c for c in mean_columns if c in data.columns]
    industry_means = data.groupby("INDUSTRY")[valid_cols].mean()

    result_data = []
    for industry in data["INDUSTRY"].unique():
        industry_data = data[data["INDUSTRY"] == industry]
        result_data.append(industry_data)

        mean_row = {"JOB TITLE": f"MEAN ({industry})", "INDUSTRY": industry}
        for col in data.columns:
            if col in valid_cols and industry in industry_means.index:
                mean_row[col] = round(industry_means.loc[industry, col], 2)
            elif col not in ["JOB TITLE", "INDUSTRY"]:
                mean_row[col] = ""
        result_data.append(pd.DataFrame([mean_row]))

    return pd.concat(result_data, ignore_index=True)

def load_data(transformed_data, file_path):
    transformed_data.to_csv(file_path, index=False)
    return pd.read_csv(file_path)

@st.cache_data(show_spinner="Running ETL pipeline…")
def get_past_data():
    raw = extract_data(PAST_RAW_PATH)
    transformed = transform_past_data(raw)
    return load_data(transformed, PAST_CLEAN_PATH)

@st.cache_data(show_spinner="Running ETL pipeline…")
def get_new_data():
    raw = extract_data(NEW_RAW_PATH)
    transformed = transform_new_data(raw)
    return load_data(transformed, NEW_CLEAN_PATH)

def linear_forecast(x1, y1, x2, y2, x_pred):
    if x2 == x1:
        return y1
    slope = (y2 - y1) / (x2 - x1)
    return max(0, y1 + slope * (x_pred - x1))

def strip_mean_new(df):
    return df[~df["JOB TITLE"].astype(str).str.startswith("MEAN")].copy()

def strip_mean_past(df):
    return df[~df["JOB_ROLE"].astype(str).str.startswith("MEAN")].copy()

INDUSTRIES = ["Education", "Finance", "Healthcare", "IT",
              "Manufacturing", "Retail", "Transportation"]

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#333",
    "grid.linewidth":   0.5,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#444",
})

def new_fig(w=10, h=5):
    return plt.subplots(figsize=(w, h))

with st.sidebar:
    st.title(" Skills Predictor")
    st.divider()

    page = st.radio("Navigate", [
        " Overview",
        " Trend Analysis",
        " Skill Forecast",
        " Automation Risk",
        " Salary Insights",
        " Personal Advisor",
    ])

    st.divider()
    st.header("Filters")
    selected_industry = st.selectbox("Industry", INDUSTRIES)
    selected_location = st.selectbox("Location", [
        "All", "USA", "UK", "India", "Australia",
        "Germany", "Canada", "China", "Brazil", "Japan",
    ])

    st.divider()
    with st.expander(" About You"):
        user_name   = st.text_input("Your Name")
        user_age    = st.slider("Age", 18, 65, 25)
        user_gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
        user_exp    = st.slider("Years of Experience", 0, 40, 3)
        user_edu    = st.selectbox("Education Level", [
            "High School", "Associate Degree",
            "Bachelor's Degree", "Master's Degree", "PhD",
        ])

try:
    past_df = get_past_data()
    new_df  = get_new_data()
except FileNotFoundError as e:
    st.error(f" Data file not found: {e}\n\nPlace all CSV files inside a `data/` folder.")
    st.stop()

NUM_COLS_NEW = [
    "MEDIAN SALARY (USD)", "OPENING GROWTH (%)", "AUTOMATION RISK (%)",
    "JOB OPENINGS (2024)", "PROJECTED OPENINGS (2030)",
    "REMOTE WORK RATIO (%)", "GENDER DIVERSITY (%)", "EXPERIENCE REQUIRED (YEARS)",
]
NUM_COLS_PAST = [
    "AUTOMATION_RISK_PERCENT", "AI_REPLACEMENT_SCORE", "SKILL_GAP_INDEX",
    "SALARY_BEFORE_USD", "SALARY_AFTER_USD", "SALARY_CHANGE_PERCENT",
    "SKILL_DEMAND_GROWTH_PERCENT", "REMOTE_FEASIBILITY_SCORE",
    "AI_ADOPTION_LEVEL", "EDUCATION_REQUIREMENT_LEVEL", "YEAR",
]

def filter_new(industry=None, location=None):
    df = strip_mean_new(new_df)
    if industry:
        df = df[df["INDUSTRY"] == industry]
    if location and location != "All" and "LOCATION" in df.columns:
        df = df[df["LOCATION"] == location]
    for col in NUM_COLS_NEW:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def filter_past(industry=None):
    df = strip_mean_past(past_df)
    if industry:
        df = df[df["INDUSTRY"] == industry]
    for col in NUM_COLS_PAST:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

if page == " Overview":
    st.title(" SkiDIP")
    st.markdown(
        "> **PS10 – HACK KRMU 5** | Predicting high demand skills over the next 3–5 years "
        "using job market data, salary trends, automation risk, and industry projections"
    )

    nd = filter_new(selected_industry, selected_location)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Median Salary",   f"${nd['MEDIAN SALARY (USD)'].mean():,.0f}")
    col2.metric("Avg Opening Growth",  f"{nd['OPENING GROWTH (%)'].mean():.1f}%")
    col3.metric("Avg Automation Risk", f"{nd['AUTOMATION RISK (%)'].mean():.1f}%")
    col4.metric("Job Openings (2024)", f"{int(nd['JOB OPENINGS (2024)'].sum()):,}")
    col5.metric("Projected (2030)",    f"{int(nd['PROJECTED OPENINGS (2030)'].sum()):,}")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f" Top 10 Fastest Growing Jobs — {selected_industry}")
        top_growth = (
            nd.groupby("JOB TITLE")["OPENING GROWTH (%)"]
            .mean().nlargest(10).sort_values()
        )
        fig, ax = new_fig(7, 5)
        bars = ax.barh(top_growth.index, top_growth.values, color='cyan')
        ax.set_xlabel("Opening Growth (%)")
        ax.set_title("Opening Growth by Job Title", color="#e0e0e0")
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, top_growth.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8, color="#ccc")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader(" Projected Openings (2030) by Industry")
        all_nd = strip_mean_new(new_df)
        all_nd["PROJECTED OPENINGS (2030)"] = pd.to_numeric(
            all_nd["PROJECTED OPENINGS (2030)"], errors="coerce")
        ind_proj = all_nd.groupby("INDUSTRY")["PROJECTED OPENINGS (2030)"].mean().sort_values()
        fig, ax = new_fig(7, 5)
        ax.barh(ind_proj.index, ind_proj.values, color='violet')
        ax.set_xlabel("Avg Projected Openings (2030)")
        ax.set_title("By Industry", color="#e0e0e0")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


    with st.expander(" Raw Data Preview"):
        st.dataframe(nd.head(50), use_container_width=True)

elif page == " Trend Analysis":
    st.title(" Historical Trend Analysis")
    pd_ = filter_past(selected_industry)

    st.subheader(f"Skill Demand Growth % Over Years — {selected_industry}")
    if "YEAR" in pd_.columns and "SKILL_DEMAND_GROWTH_PERCENT" in pd_.columns:
        yearly = pd_.groupby("YEAR")["SKILL_DEMAND_GROWTH_PERCENT"].mean().dropna()
        fig, ax = new_fig(10, 4)
        ax.plot(yearly.index, yearly.values, marker="o", color="#4C8BF5",
                linewidth=2, markersize=6)
        ax.fill_between(yearly.index, yearly.values, alpha=0.15, color="#4C8BF5")
        ax.set_xlabel("Year")
        ax.set_ylabel("Avg Skill Demand Growth (%)")
        ax.set_title("Skill Demand Growth Over Time", color="#e0e0e0")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader(" Salary Before vs After (Top 10 Roles)")
        if "SALARY_BEFORE_USD" in pd_.columns and "SALARY_AFTER_USD" in pd_.columns:
            sal_avg = (
                pd_.groupby("JOB_ROLE")[["SALARY_BEFORE_USD", "SALARY_AFTER_USD"]]
                .mean().nlargest(10, "SALARY_BEFORE_USD")
            )
            fig, ax = new_fig(7, 5)
            x = np.arange(len(sal_avg))
            w = 0.35
            ax.bar(x - w/2, sal_avg["SALARY_BEFORE_USD"], w, label="Before", color="#4C8BF5")
            ax.bar(x + w/2, sal_avg["SALARY_AFTER_USD"],  w, label="After",  color="#F5A623")
            ax.set_xticks(x)
            ax.set_xticklabels(sal_avg.index, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Salary (USD)")
            ax.legend()
            ax.set_title("Salary Comparison", color="#e0e0e0")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c2:
        st.subheader(" AI Adoption Level (Top 10 Roles)")
        if "AI_ADOPTION_LEVEL" in pd_.columns:
            ai_adopt = (
                pd_.groupby("JOB_ROLE")["AI_ADOPTION_LEVEL"]
                .mean().nlargest(10).sort_values()
            )
            fig, ax = new_fig(7, 5)
            colors = cm.Reds(np.linspace(0.4, 0.9, len(ai_adopt)))
            ax.barh(ai_adopt.index, ai_adopt.values, color=colors)
            ax.set_xlabel("Avg AI Adoption Level")
            ax.set_title("AI Adoption by Role", color="#e0e0e0")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.subheader(" Remote Feasibility Score Distribution")
    if "REMOTE_FEASIBILITY_SCORE" in pd_.columns:
        fig, ax = new_fig(10, 4)
        ax.hist(pd_["REMOTE_FEASIBILITY_SCORE"].dropna(), bins=30,
                color="#26C6DA", edgecolor="#0e1117")
        ax.set_xlabel("Remote Feasibility Score")
        ax.set_ylabel("Count")
        ax.set_title("Remote Feasibility Distribution", color="#e0e0e0")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with st.expander(" Historical Data Table"):
        st.dataframe(pd_.head(100), use_container_width=True)

elif page == " Skill Forecast":
    st.title(" Skill Demand Forecasting (2024–2030)")
    st.markdown(
        f"**Industry:** `{selected_industry}` | "
        "Linear interpolation between 2024 baseline and 2030 projections."
    )

    nd = filter_new(selected_industry, selected_location)
    nd = nd.dropna(subset=["JOB OPENINGS (2024)", "PROJECTED OPENINGS (2030)"])

    agg = (
        nd.groupby("JOB TITLE")[["JOB OPENINGS (2024)", "PROJECTED OPENINGS (2030)"]]
        .mean()
    )
    agg["NET GROWTH"] = agg["PROJECTED OPENINGS (2030)"] - agg["JOB OPENINGS (2024)"]

    n_titles = st.slider("Top N job titles to display", 5, 20, 10)
    top_titles = agg.nlargest(n_titles, "NET GROWTH").index.tolist()

    years = list(range(2024, 2031))
    fig, ax = new_fig(12, 6)
    cmap_fn = cm.get_cmap("tab20", len(top_titles))

    for i, title in enumerate(top_titles):
        y2024 = agg.loc[title, "JOB OPENINGS (2024)"]
        y2030 = agg.loc[title, "PROJECTED OPENINGS (2030)"]
        vals  = [linear_forecast(2024, y2024, 2030, y2030, yr) for yr in years]
        ax.plot(years, vals, marker="o", markersize=4,
                color=cmap_fn(i), linewidth=1.8, label=title)

    ax.set_xlabel("Year")
    ax.set_ylabel("Projected Job Openings")
    ax.set_title(f"Job Opening Forecast 2024–2030 | {selected_industry}", color="#e0e0e0")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader(" Top Growing Roles (2024 → 2030)")
    display = (
        agg.nlargest(20, "NET GROWTH")
        .rename(columns={
            "JOB OPENINGS (2024)": "Openings (2024)",
            "PROJECTED OPENINGS (2030)": "Openings (2030)",
            "NET GROWTH": "Net Growth",
        })
        .reset_index()
        .round(0)
    )
    st.dataframe(display, use_container_width=True)

elif page == " Automation Risk":
    st.title(" Automation Risk Analysis")
    nd = filter_new(selected_industry, selected_location)
    risk_avg = nd.groupby("JOB TITLE")["AUTOMATION RISK (%)"].mean().dropna()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(" 15 Most At-Risk Roles")
        top_risk = risk_avg.nlargest(15).sort_values()
        fig, ax = new_fig(7, 6)
        colors = cm.OrRd(np.linspace(0.4, 0.9, len(top_risk)))
        ax.barh(top_risk.index, top_risk.values, color=colors)
        ax.set_xlabel("Automation Risk (%)")
        ax.set_title("Highest Risk", color="#e0e0e0")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("15 Safest Roles")
        low_risk = risk_avg.nsmallest(15).sort_values(ascending=False)
        fig, ax = new_fig(7, 6)
        colors = cm.Greens(np.linspace(0.5, 0.9, len(low_risk)))
        ax.barh(low_risk.index, low_risk.values, color=colors)
        ax.set_xlabel("Automation Risk (%)")
        ax.set_title("Lowest Risk", color="#e0e0e0")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader(" Avg Automation Risk by Industry")
    all_nd = strip_mean_new(new_df)
    all_nd["AUTOMATION RISK (%)"] = pd.to_numeric(all_nd["AUTOMATION RISK (%)"], errors="coerce")
    ind_risk = all_nd.groupby("INDUSTRY")["AUTOMATION RISK (%)"].mean().sort_values()
    fig, ax = new_fig(10, 4)
    colors = cm.RdYlGn_r(np.linspace(0.1, 0.9, len(ind_risk)))
    bars = ax.barh(ind_risk.index, ind_risk.values, color=colors)
    for bar, val in zip(bars, ind_risk.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Avg Automation Risk (%)")
    ax.set_title("Automation Risk by Industry", color="#e0e0e0")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader(" Automation Risk vs Opening Growth (Scatter)")
    scatter = nd[["JOB TITLE", "AUTOMATION RISK (%)", "OPENING GROWTH (%)"]].dropna()
    fig, ax = new_fig(10, 5)
    sc = ax.scatter(
        scatter["AUTOMATION RISK (%)"], scatter["OPENING GROWTH (%)"],
        c=scatter["OPENING GROWTH (%)"], cmap="RdYlGn",
        alpha=0.6, s=25, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Opening Growth (%)")
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Automation Risk (%)")
    ax.set_ylabel("Opening Growth (%)")
    ax.set_title(f"Risk vs Growth — {selected_industry}", color="#e0e0e0")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

elif page == " Salary Insights":
    st.title(" Salary Insights")
    nd  = filter_new(selected_industry, selected_location)
    pd_ = filter_past(selected_industry)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Top 10 Highest Paying Roles — {selected_industry}")
        top_sal = nd.groupby("JOB TITLE")["MEDIAN SALARY (USD)"].mean().nlargest(10).sort_values()
        fig, ax = new_fig(7, 5)
        colors = cm.Blues(np.linspace(0.4, 0.9, len(top_sal)))
        ax.barh(top_sal.index, top_sal.values, color=colors)
        ax.set_xlabel("Median Salary (USD)")
        ax.set_title("Highest Paying Roles", color="#e0e0e0")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Salary by Education Level")
        if "REQUIRED EDUCATION" in nd.columns:
            edu_sal = nd.groupby("REQUIRED EDUCATION")["MEDIAN SALARY (USD)"].mean().sort_values()
            fig, ax = new_fig(7, 5)
            colors = cm.Purples(np.linspace(0.4, 0.9, len(edu_sal)))
            ax.barh(edu_sal.index, edu_sal.values, color=colors)
            ax.set_xlabel("Avg Median Salary (USD)")
            ax.set_title("Salary by Education Level", color="#e0e0e0")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.subheader(" Salary Change % — Top Gainers & Losers (Historical)")
    if "SALARY_CHANGE_PERCENT" in pd_.columns:
        sal_change = pd_.groupby("JOB_ROLE")["SALARY_CHANGE_PERCENT"].mean().dropna()
        combined = pd.concat([sal_change.nlargest(8), sal_change.nsmallest(8)]).sort_values()
        fig, ax = new_fig(10, 6)
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in combined.values]
        ax.barh(combined.index, combined.values, color=colors)
        ax.axvline(0, color="#aaa", linewidth=0.8)
        ax.set_xlabel("Salary Change (%)")
        ax.set_title("Top Salary Gainers & Losers", color="#e0e0e0")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader(" Avg Salary Heatmap: Industry × AI Impact Level")
    all_nd = strip_mean_new(new_df)
    all_nd["MEDIAN SALARY (USD)"] = pd.to_numeric(all_nd["MEDIAN SALARY (USD)"], errors="coerce")
    if "AI IMPACT LEVEL" in all_nd.columns:
        hm = (
            all_nd.groupby(["INDUSTRY", "AI IMPACT LEVEL"])["MEDIAN SALARY (USD)"]
            .mean().unstack(fill_value=0)
        )
        fig, ax = new_fig(10, 5)
        im = ax.imshow(hm.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(hm.columns)))
        ax.set_xticklabels(hm.columns, color="#e0e0e0")
        ax.set_yticks(range(len(hm.index)))
        ax.set_yticklabels(hm.index, color="#e0e0e0")
        plt.colorbar(im, ax=ax, label="Avg Salary (USD)")
        ax.set_title("Salary Heatmap: Industry × AI Impact", color="#e0e0e0")
        for i in range(len(hm.index)):
            for j in range(len(hm.columns)):
                ax.text(j, i, f"${hm.values[i, j]:,.0f}",
                        ha="center", va="center", fontsize=7, color="black")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

elif page == " Personal Advisor":
    st.title(" Personalised Career & Skill Advisor")

    greeting = f"Hello, **{user_name}**! " if user_name else " Welcome! Enter your name in the sidebar."
    st.markdown(greeting)
    st.markdown(
        f"> **Age:** {user_age} | **Experience:** {user_exp} yrs | "
        f"**Education:** {user_edu} | **Industry:** {selected_industry} | **Location:** {selected_location}"
    )
    st.divider()

    nd = filter_new(selected_industry, selected_location)

    if "REQUIRED EDUCATION" in nd.columns:
        nd_filtered = nd[nd["REQUIRED EDUCATION"] == user_edu]
        if nd_filtered.empty:
            nd_filtered = nd
            st.info("No exact education match — showing all roles in this industry.")
    else:
        nd_filtered = nd

    st.subheader(" Recommended Job Roles for You")
    if not nd_filtered.empty:
        recs = (
            nd_filtered.groupby("JOB TITLE")
            .agg(
                Growth=("OPENING GROWTH (%)", "mean"),
                Salary=("MEDIAN SALARY (USD)", "mean"),
                AutoRisk=("AUTOMATION RISK (%)", "mean"),
                Remote=("REMOTE WORK RATIO (%)", "mean"),
            )
            .dropna()
            .sort_values("Growth", ascending=False)
            .head(10)
            .reset_index()
            .round(2)
        )
        recs.columns = ["Job Title", "Growth %", "Avg Salary (USD)", "Auto Risk %", "Remote %"]

        fig, ax = new_fig(10, 5)
        colors = cm.RdYlGn(np.linspace(0.2, 0.9, len(recs)))
        bars = ax.barh(recs["Job Title"], recs["Growth %"], color=colors)
        ax.set_xlabel("Opening Growth (%)")
        ax.set_title(f"Top Roles for You — {selected_industry}", color="#e0e0e0")
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, recs["Growth %"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8, color="#ccc")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(recs, use_container_width=True)

        if not recs.empty:
            top = recs.iloc[0]
            st.success(
                f" **Top Pick:** *{top['Job Title']}* — "
                f"{top['Growth %']:.1f}% growth | "
                f"${top['Avg Salary (USD)']:,.0f} avg salary | "
                f"{top['Auto Risk %']:.1f}% automation risk"
            )

    st.divider()
    st.subheader(" Most Future-Proof Roles (High Growth + Low Risk)")
    emerging = (
        nd.groupby("JOB TITLE")
        .agg(growth=("OPENING GROWTH (%)", "mean"), risk=("AUTOMATION RISK (%)", "mean"))
        .dropna()
    )
    emerging["score"] = emerging["growth"] - emerging["risk"]
    top_ep = emerging.nlargest(10, "score").sort_values("score")

    fig, ax = new_fig(10, 5)
    colors = cm.cool(np.linspace(0.2, 0.9, len(top_ep)))
    ax.barh(top_ep.index, top_ep["score"], color=colors)
    ax.set_xlabel("Future-Proof Score  (Growth % − Automation Risk %)")
    ax.set_title("Most Future-Proof Roles", color="#e0e0e0")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader(" Job Openings by Location")
    if "LOCATION" in nd.columns:
        loc_data = (
            filter_new(selected_industry)
            .groupby("LOCATION")["JOB OPENINGS (2024)"].sum().sort_values(ascending=False)
        )
        fig, ax = new_fig(10, 4)
        colors = cm.plasma(np.linspace(0.2, 0.8, len(loc_data)))
        ax.bar(loc_data.index, loc_data.values, color=colors)
        ax.set_xlabel("Location")
        ax.set_ylabel("Total Job Openings (2024)")
        ax.set_title(f"Job Openings by Location — {selected_industry}", color="#e0e0e0")
        plt.xticks(rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()