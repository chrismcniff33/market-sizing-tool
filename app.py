%%writefile app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import re
import json
from openai import OpenAI
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Custom Category Market Sizing", layout="wide", page_icon="📐")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .insight-box { background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; margin-bottom: 20px; }
    .instruction-text { font-size: 14px; color: #555; margin-bottom: 20px; }
    .kpi-card { background-color: #fff; padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center; }
    div.stButton > button { background-color: #007bff; color: white; border-radius: 8px; width: 100%; height: 3em; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def clean_currency(value):
    """Removes symbols and converts string currency to float."""
    if pd.isna(value): return 0.0
    val_str = str(value)
    clean_val = re.sub(r'[^\d.]', '', val_str)
    try: return float(clean_val)
    except ValueError: return 0.0

def estimate_cost(num_rows):
    """Estimates OpenAI API cost for gpt-4o-mini."""
    input_tokens = num_rows * 150 # Increased buffer for better accuracy
    output_tokens = num_rows * 50
    cost_input = (input_tokens / 1_000_000) * 0.15
    cost_output = (output_tokens / 1_000_000) * 0.60
    return cost_input + cost_output

# --- AI FUNCTIONS (CACHED) ---

@st.cache_data(show_spinner=False, ttl=3600)
def check_euromonitor(_client, definition, examples):
    """
    Checks if a custom definition exists in Euromonitor's standard taxonomy.
    Cached for 1 hour to prevent re-running same check.
    """
    system_prompt = """
    You are a Senior Market Research Analyst and expert in Euromonitor Passport's taxonomy.
    Determine if a custom user category falls within Euromonitor's standard definitions.
    Return JSON: {"status": "Fully Covered"|"Partially Covered"|"Not Covered", "euromonitor_categories": [list strings], "explanation": "string"}
    """
    user_prompt = f"Definition: '{definition}'\nExamples: {examples}"
    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"status": "Error", "explanation": str(e), "euromonitor_categories": []}

@st.cache_data(show_spinner=False, ttl=3600)
def classify_sku_batch(_client, category_def, skus_data_json):
    """
    Classifies a batch of SKUs.
    Input 'skus_data_json' must be a JSON string to be hashable for caching.
    """
    skus_data = json.loads(skus_data_json)

    system_prompt = f"""
    You are a strict taxonomy expert.
    Category Definition: "{category_def}"
    Analyze each SKU. Return JSON with key "results": list of {{ "id": int, "is_match": bool, "rationale": string }}.
    """
    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(skus_data)}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"results": []}

def generate_summary_insights(client, category_def, top_brands, total_skus, matched_skus, market_val):
    """Generates qualitative strategic insights (Not cached as context changes frequently)."""
    system_prompt = "You are a Senior Market Strategy Consultant. Synthesize the data provided into 5 sharp, strategic insights."
    user_prompt = f"""
    Context:
    - Custom Category: "{category_def}"
    - Total Market Value: {market_val}
    - Total SKUs Analyzed: {total_skus}
    - SKUs Matching Definition: {matched_skus}
    - Top Performing Brands: {top_brands}

    Task: Write 4-6 bullet points covering:
    1. Who is leading (Incumbents vs Disruptors).
    2. Data quality rationale (how many SKUs matched).
    3. Any observation on the 'Long Tail' (Others).
    Format: Return JSON object with key "insights" containing a list of strings.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get("insights", [])
    except:
        return ["Could not generate qualitative insights."]

# --- MAIN APP LAYOUT ---

st.title("📐 Custom Category Market Sizing")
st.markdown("""
Use **Euromonitor Passport** and **SKU datasets** to quickly and accurately size categories
in line with how you view the universe you are playing in.
""")
st.divider()

# Session State Initialization
if 'df_p' not in st.session_state: st.session_state.df_p = None
if 'df_s' not in st.session_state: st.session_state.df_s = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'alignment_check' not in st.session_state: st.session_state.alignment_check = None
if 'qual_insights' not in st.session_state: st.session_state.qual_insights = None

# Sidebar
with st.sidebar:
    st.header("🔑 API Credentials")
    oa_key = st.text_input("OpenAI API Key", type="password")
    tavily_key = st.text_input("Tavily API Key (Optional)", type="password")

    st.divider()
    st.header("⚙️ Processing Settings")
    process_all = st.checkbox("Process Entire File", value=False)
    process_limit = 0 if process_all else st.slider("Test Limit (Rows)", 10, 500, 50)
    st.caption("v3.1 | Caching & Exports Active")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Definition Alignment", "2️⃣ Passport Data", "3️⃣ Upload SKUs", "📊 Insights & Rationale"])

# TAB 1: DEFINITION
with tab1:
    st.subheader("Define & Validate Category")
    st.markdown('<p class="instruction-text">Start by defining your custom category in plain English. The AI will cross-reference this with Euromonitor\'s standard taxonomy to determine where your market sits today.</p>', unsafe_allow_html=True)

    col_def, col_ex = st.columns([3, 2])
    with col_def:
        category_definition = st.text_area("Category Definition:", height=150, key="cat_def_input", placeholder="e.g. Sulfate-free shampoo for curly hair...")
    with col_ex:
        st.write("**Reference Examples (Optional)**")
        ex1 = st.text_input("URL 1")
        ex2 = st.text_input("URL 2")
        ex3 = st.text_input("URL 3")

    if st.button("🔍 Validate Definition"):
        if not oa_key: st.error("Please enter your OpenAI API Key in the sidebar.")
        else:
            client = OpenAI(api_key=oa_key) # Client is not cached, passed to cached func
            with st.spinner("Consulting Euromonitor Taxonomy..."):
                st.session_state.alignment_check = check_euromonitor(client, category_definition, [ex1, ex2, ex3])

    if st.session_state.alignment_check:
        res = st.session_state.alignment_check
        status = res.get('status', 'Error')
        color = "#d4edda" if "Fully" in status else "#fff3cd" if "Partially" in status else "#f8d7da"
        st.markdown(f'<div style="background-color:{color}; padding:15px; border-radius:5px; margin-top:20px;"><b>{status}:</b> {res["explanation"]}</div>', unsafe_allow_html=True)

# TAB 2: PASSPORT
with tab2:
    st.subheader("Import Market Sales Data")
    st.markdown('<p class="instruction-text">Upload your <b>Euromonitor Passport</b> export here. Ensure the file contains Brand Name and Retail Value RSP columns. This will be used as the "Control Total" for the market.</p>', unsafe_allow_html=True)

    p_file = st.file_uploader("Upload Passport CSV", type="csv")
    if p_file:
        df_p_raw = pd.read_csv(p_file)
        st.session_state.df_p = df_p_raw
        c1, c2 = st.columns(2)
        with c1: brand_col_p = st.selectbox("Select Brand Column", df_p_raw.columns, key='bp')
        with c2: sales_col_p = st.selectbox("Select Sales Column (Local Currency)", df_p_raw.columns, key='sp')
        df_p_raw['CLEANED_SALES'] = df_p_raw[sales_col_p].apply(clean_currency)
        st.session_state.mapped_p = (brand_col_p, 'CLEANED_SALES')
        st.dataframe(df_p_raw.head(3), use_container_width=True)

# TAB 3: SKUS
with tab3:
    st.subheader("Import SKU Attributes")
    st.markdown('<p class="instruction-text">Upload your SKU-level dataset. The AI will read the Product Name, Description, and URL to determine if each SKU belongs in your custom category.</p>', unsafe_allow_html=True)

    s_file = st.file_uploader("Upload SKU CSV", type="csv")
    if s_file:
        df_s_raw = pd.read_csv(s_file)
        st.session_state.df_s = df_s_raw

        c1, c2, c3 = st.columns(3)
        with c1: brand_col_s = st.selectbox("SKU Brand Column", df_s_raw.columns, key='bs')
        with c2: name_col_s = st.selectbox("SKU Name Column", df_s_raw.columns, key='ns')
        with c3: retailer_col_s = st.selectbox("Retailer Column (Optional)", ["None"] + list(df_s_raw.columns), key='rs')

        c4, c5, c6 = st.columns(3)
        with c4: price_col_s = st.selectbox("Price Column (Optional)", ["None"] + list(df_s_raw.columns), key='ps')
        with c5: url_col_s = st.selectbox("URL Column (Optional)", ["None"] + list(df_s_raw.columns), key='us')
        with c6: desc_col_s = st.selectbox("Description Column (Optional)", ["None"] + list(df_s_raw.columns), key='ds')

        st.session_state.mapped_s = (brand_col_s, name_col_s, retailer_col_s, price_col_s, url_col_s, desc_col_s)
        st.dataframe(df_s_raw.head(3), use_container_width=True)

# TAB 4: INSIGHTS
with tab4:
    st.subheader("Market Sizing & Analysis")
    st.markdown('<p class="instruction-text">Review the AI-generated market size. This dashboard aggregates SKU decisions back up to Brand level and re-allocates Passport sales data.</p>', unsafe_allow_html=True)

    if st.session_state.get('cat_def_input') and st.session_state.df_p is not None and st.session_state.df_s is not None:

        # --- EXECUTION BUTTON ---
        if st.session_state.processed_data is None:
            df_s = st.session_state.df_s
            rows = len(df_s) if process_all else min(process_limit, len(df_s))
            est = estimate_cost(rows)
            st.info(f"Ready to process {rows:,} SKUs. Est Cost: ${est:.4f}")

            if st.button("🚀 Run Market Sizing"):
                if not oa_key: st.stop()
                client = OpenAI(api_key=oa_key)

                # DATA PREP
                df_subset = df_s.head(rows).copy()
                b_s, n_s, r_s, p_s, u_s, d_s = st.session_state.mapped_s

                # RUN AI LOOP
                progress = st.progress(0, text="Analyzing...")
                results_map = {}
                batch_size = 10

                for i in range(0, rows, batch_size):
                    batch = df_subset.iloc[i : i+batch_size]
                    payload = []
                    for idx, row in batch.iterrows():
                        txt = f"Product: {row[n_s]}"
                        if d_s != "None": txt += f", Desc: {str(row[d_s])[:150]}"
                        if u_s != "None": txt += f", URL: {str(row[u_s])}"
                        payload.append({"id": idx, "text": txt})

                    # Call CACHED Function
                    resp = classify_sku_batch(client, st.session_state.cat_def_input, json.dumps(payload))

                    if "results" in resp:
                        for item in resp["results"]: results_map[item["id"]] = item
                    progress.progress(min((i+batch_size)/rows, 1.0))

                # MAPPING RESULTS
                df_subset['is_match'] = df_subset.index.map(lambda x: results_map.get(x, {}).get('is_match', False))
                df_subset['rationale'] = df_subset.index.map(lambda x: results_map.get(x, {}).get('rationale', "Failed"))
                st.session_state.processed_data = df_subset

                # CALCULATION ENGINE
                df_results = df_subset
                df_p = st.session_state.df_p
                b_p, s_p = st.session_state.mapped_p

                # 1. Brand Stats
                brand_stats = df_results.groupby(b_s).agg(
                    total=(b_s, 'count'),
                    matched=('is_match', 'sum')
                ).reset_index()
                brand_stats['ratio'] = brand_stats['matched'] / brand_stats['total']

                # 2. Others Logic
                known_brands = df_p[b_p].unique()
                others_stats = brand_stats[~brand_stats[b_s].isin(known_brands)]
                others_ratio = others_stats['matched'].sum() / others_stats['total'].sum() if not others_stats.empty else 0

                # 3. Merge & Apply
                df_final = pd.merge(df_p, brand_stats, left_on=b_p, right_on=b_s, how='left')
                df_final['ratio'] = df_final['ratio'].fillna(0)
                df_final.loc[df_final[b_p].str.contains("Other", case=False, na=False), 'ratio'] = others_ratio
                df_final['custom_sales'] = df_final[s_p] * df_final['ratio']

                st.session_state.df_final = df_final

                # 4. Generate Qualitative Insights
                top_5_str = ", ".join(df_final.sort_values('custom_sales', ascending=False).head(5)[b_p].tolist())
                m_val = f"{df_final['custom_sales'].sum()/1e6:.1f}M"
                st.session_state.qual_insights = generate_summary_insights(
                    client, st.session_state.cat_def_input, top_5_str, rows, df_subset['is_match'].sum(), m_val
                )
                st.rerun()

    # --- DASHBOARD DISPLAY ---
    if st.session_state.processed_data is not None:
        df_final = st.session_state.df_final
        df_results = st.session_state.processed_data
        b_p = st.session_state.mapped_p[0]
        p_s = st.session_state.mapped_s[3] # Price column

        total_custom_sales = df_final['custom_sales'].sum()
        matched_skus = df_results['is_match'].sum()
        total_skus = len(df_results)

        # 1. KPI HIGHLIGHT BOXES
        k1, k2, k3 = st.columns(3)
        def kpi(label, value, subtext):
            return f"""<div class="kpi-card"><div style="font-size: 14px; color: #666;">{label}</div><div style="font-size: 32px; font-weight: bold; color: #007bff; margin: 5px 0;">{value}</div><div style="font-size: 12px; color: #888;">{subtext}</div></div>"""

        k1.markdown(kpi("Custom Market Size", f"{total_custom_sales/1e6:,.1f}M", "Total Value (Local Currency)"), unsafe_allow_html=True)
        k2.markdown(kpi("SKU Alignment", f"{matched_skus}", f"out of {total_skus} analyzed"), unsafe_allow_html=True)
        k3.markdown(kpi("Portfolio Share", f"{(matched_skus/total_skus)*100:.1f}%", "SKUs matching definition"), unsafe_allow_html=True)

        st.markdown("---")

        # 2. SPLIT VIEW: Donut & Insights
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.write("### 🍩 Category Share by Brand")
            chart_df = df_final[df_final['custom_sales'] > 0].sort_values('custom_sales', ascending=False)
            fig = px.pie(chart_df, values='custom_sales', names=b_p, hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism, hover_data={'custom_sales': ':,.0f'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.write("### 💡 Strategic Rationale")
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            if st.session_state.qual_insights:
                for insight in st.session_state.qual_insights: st.markdown(f"• {insight}")
            else: st.write("Generating insights...")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # 3. PRICE ARCHITECTURE (Optional)
        if p_s != "None":
            st.write("### 🏷️ Price Architecture Strategy")
            st.caption("Compare the pricing power of your custom category vs. the rest of the portfolio.")

            # Clean price column
            df_results['clean_price'] = df_results[p_s].apply(clean_currency)
            df_results['Segment'] = df_results['is_match'].apply(lambda x: "Custom Category" if x else "Rest of Portfolio")
            price_chart_df = df_results[df_results['clean_price'] > 0]

            fig_price = px.box(
                price_chart_df,
                x='Segment', y='clean_price', color='Segment', points="outliers",
                color_discrete_map={"Custom Category": "#007bff", "Rest of Portfolio": "#d3d3d3"}
            )
            st.plotly_chart(fig_price, use_container_width=True)

        # 4. VERIFIED EXAMPLES & EXPORT
        st.write("### ✅ Verified Product Examples")
        matches = df_results[df_results['is_match'] == True]

        if not matches.empty:
            b_s, n_s, r_s, p_s, u_s, _ = st.session_state.mapped_s
            example_table = pd.DataFrame()
            example_table['Brand'] = matches[b_s]
            example_table['Product Name'] = matches[n_s]
            if r_s != "None": example_table['Retailer'] = matches[r_s]
            if p_s != "None": example_table['Price'] = matches[p_s]
            if u_s != "None": example_table['URL'] = matches[u_s]

            st.dataframe(example_table.head(10), use_container_width=True, column_config={"URL": st.column_config.LinkColumn("Product Link")})

            # EXPORT BUTTON
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Classification (.csv)",
                data=csv_data,
                file_name="custom_category_skus.csv",
                mime="text/csv"
            )
        else:
            st.warning("No SKUs matched your definition.")
