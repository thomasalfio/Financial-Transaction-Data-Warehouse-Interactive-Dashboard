import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image
import base64
import io
from lida import Manager, TextGenerationConfig, llm

# Global constants used across the file
QUARTER_ORDER = ['Q1', 'Q2', 'Q3', 'Q4']
TRANSACTION_COLORS = {
    'BUY': '#2E8B57',
    'SELL': '#DC143C',
    'DIVIDEND': '#FF8C00'
}

# Preparation
st.set_page_config(
    page_title="Financial Transactions Dashboard",
    page_icon="📊",
    layout="wide"
)

# Function to sort quarters based on Q order
def sort_quarters(data, quarter_column='quarter'):
    """Sort data by quarter using the global quarter order"""
    data[quarter_column] = pd.Categorical(data[quarter_column], categories=QUARTER_ORDER, ordered=True)
    return data.sort_values(quarter_column)

# handle visulazation (in particular for the lida part)
def display_image(base64_string):
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes))
    st.image(img, use_container_width=True)

@st.cache_data
def load_data():
    merged_data = pd.read_csv('transactions_merged.csv')
    return merged_data

# Query function with quarter filtering
def query_data(data, selected_quarters=None):
    if selected_quarters:
        filtered_data = data[data['quarter'].isin(selected_quarters)]
    else:
        filtered_data = data
    return filtered_data

# First page used
def main_dashboard():
    st.title("📊 Financial Transactions Dashboard")
    st.markdown("### Interactive Analysis of Stock Trading Activities")
    
    # Load data (of the ipynb file)
    data = load_data()

    # Quarter picker section
    st.sidebar.header("📊 Quarter Selection")
    
    available_quarters = sorted(data['quarter'].unique())
    
    quarter_range = st.sidebar.select_slider(
        "Select Quarter Range:",
        options=available_quarters,
        value=(available_quarters[0], available_quarters[-1]),
        key="quarter_range"
    )
    
    start_quarter_idx = available_quarters.index(quarter_range[0])
    end_quarter_idx = available_quarters.index(quarter_range[1])
    final_quarters = available_quarters[start_quarter_idx:end_quarter_idx + 1]
    
    filtered_data = query_data(data, selected_quarters=final_quarters)
    filter_display = f"Quarters: {', '.join(final_quarters)}"
    
    st.sidebar.header("🎯 Additional Filters")
    
    available_transaction_types = sorted(data['transaction_type'].unique())
    selected_transaction_types = st.sidebar.multiselect(
        "Transaction Types",
        options=available_transaction_types,
        default=available_transaction_types,
        key="transaction_type_filter"
    )
    
    st.sidebar.header("🏢 Display Options")
    display_option = st.sidebar.radio(
        "Show companies by:",
        options=["Symbol", "Company Name"],
        key="display_option"
    )
    
    if selected_transaction_types:
        filtered_data = filtered_data[filtered_data['transaction_type'].isin(selected_transaction_types)]
    
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_transactions = len(filtered_data)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col2:
        buy_transactions = len(filtered_data[filtered_data['transaction_type'] == 'BUY'])
        st.metric("BUY Transactions", f"{buy_transactions:,}")
    
    with col3:
        sell_transactions = len(filtered_data[filtered_data['transaction_type'] == 'SELL'])
        st.metric("SELL Transactions", f"{sell_transactions:,}")
    
    with col4:
        dividend_transactions = len(filtered_data[filtered_data['transaction_type'].str.contains('DIVID', na=False)])
        st.metric("DIVIDEND Transactions", f"{dividend_transactions:,}")
    
    with col5:
        unique_symbols = filtered_data['symbol'].nunique()
        st.metric("Unique Symbols", f"{unique_symbols:,}")

    # creation of the charts
    st.markdown("---")
    st.subheader("📈 Quarterly Transaction Volume by Type")
    
    quarterly_type_data = filtered_data.groupby(['quarter', 'transaction_type']).size().reset_index(name='count')
    quarterly_type_data = sort_quarters(quarterly_type_data)
    
    fig_stacked = px.bar(
        quarterly_type_data,
        x='quarter',
        y='count',
        color='transaction_type',
        title=f"Transaction Volume by Type ({filter_display})",
        labels={'count': 'Number of Transactions', 'quarter': 'Quarter'},
        color_discrete_map=TRANSACTION_COLORS
    )
    
    fig_stacked.update_traces(texttemplate='', textposition='inside')
    fig_stacked.update_layout(
        height=550,
        legend_title_text='Transaction Type',
        xaxis_title='Quarter',
        yaxis_title='Number of Transactions',
        barmode='stack' # if you use a version of plotly < 5.0 the graph will not be stacked
    )
    st.plotly_chart(fig_stacked, use_container_width=True)
    
    # Linechart creation
    st.markdown("---")
    st.subheader("📊 Quarterly BUY+SELL Trends (Tot: Total transactions in parentheses)")
    
    quarterly_transactions = filtered_data.groupby('quarter').size().reset_index(name='total_count')
    quarterly_transactions = sort_quarters(quarterly_transactions)
    
    buy_sell_data = filtered_data[filtered_data['transaction_type'].isin(['BUY', 'SELL'])]
    buy_sell_quarterly = buy_sell_data.groupby('quarter').size().reset_index(name='buy_sell_count')
    buy_sell_quarterly = sort_quarters(buy_sell_quarterly)
    
    trend_data = quarterly_transactions.merge(buy_sell_quarterly, on='quarter', how='left')
    trend_data['buy_sell_count'] = trend_data['buy_sell_count'].fillna(0)
    trend_data['buy_sell_percentage'] = (trend_data['buy_sell_count'] / trend_data['total_count'] * 100).round(1)
    
    trend_data['prev_buy_sell'] = trend_data['buy_sell_count'].shift(1)
    trend_data['buy_sell_pct_change'] = ((trend_data['buy_sell_count'] - trend_data['prev_buy_sell']) / trend_data['prev_buy_sell'] * 100).round(1)
    
    fig_minimal = go.Figure()
    
    fig_minimal.add_trace(go.Scatter(
        x=trend_data['quarter'],
        y=trend_data['buy_sell_count'],
        mode='lines+markers',
        line=dict(width=2, color='#4A90E2'),
        marker=dict(size=8, color='#4A90E2'),
        hovertemplate='<b>%{x}</b><br>BUY+SELL: %{y:,}<br>Total: %{customdata:,}<extra></extra>',
        customdata=trend_data['total_count'],
        showlegend=False
    ))
    
    for i in range(1, len(trend_data)):
        pct_change = trend_data.iloc[i]['buy_sell_pct_change']
        if not pd.isna(pct_change):
            change_color = '#10B981' if pct_change > 0 else '#EF4444'
            
            x_mid = i - 0.5
            y_mid = (trend_data.iloc[i]['buy_sell_count'] + trend_data.iloc[i-1]['buy_sell_count']) / 2
            
            fig_minimal.add_annotation(
                x=x_mid,
                y=y_mid,
                text=f"{pct_change:+.1f}%",
                showarrow=False,
                font=dict(size=9, color=change_color, family="Inter, sans-serif"),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=change_color,
                borderwidth=1,
                yshift=10  
            )
    
    for i, row in trend_data.iterrows():
        fig_minimal.add_annotation(
            x=row['quarter'],
            y=row['buy_sell_count'],
            text=f"{row['buy_sell_count']:,}<br>(Tot: {row['total_count']:,})",
            showarrow=False,
            yshift=25,
            font=dict(size=9, color='#374151'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E5E7EB',
            borderwidth=1
        )
    
    fig_minimal.update_layout(
        xaxis_title='Quarter',
        yaxis_title='BUY+SELL Transactions',
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12, color='#374151'),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#F3F4F6',
            zeroline=False,
            tickfont=dict(size=11)
        ),
        margin=dict(l=60, r=40, t=40, b=60)
    )
    
    st.plotly_chart(fig_minimal, use_container_width=True)
    
    st.markdown("---")

    st.subheader("🏆 Top 3 Traded Companies by Transaction Type")
    
    if display_option == "Company Name":
        group_column = 'company_name'
        chart_title = "By Company Name"
        x_label = 'Company Name'
    else:
        group_column = 'symbol'
        chart_title = "By Symbol"
        x_label = 'Symbol'
    
    top_companies_total = filtered_data.groupby(group_column).size().reset_index(name='total_count')
    top_companies_total = top_companies_total.sort_values('total_count', ascending=False)
    top_companies_list = top_companies_total.head(3)[group_column].tolist()
    
    top_companies_data = filtered_data[filtered_data[group_column].isin(top_companies_list)]
    company_type_data = top_companies_data.groupby([group_column, 'transaction_type']).size().reset_index(name='count')
    
    company_totals = company_type_data.groupby(group_column)['count'].sum().reset_index()
    company_totals = company_totals.sort_values('count', ascending=False)
    company_order = company_totals[group_column].tolist()
    
    company_type_data[group_column] = pd.Categorical(company_type_data[group_column], categories=company_order, ordered=True)
    company_type_data = company_type_data.sort_values(group_column)
    
    fig_companies = px.bar(
        company_type_data,
        x=group_column,
        y='count',
        color='transaction_type',
        title=f"Top 3 Traded Companies {chart_title}",
        labels={'count': 'Number of Transactions', group_column: x_label},
        color_discrete_map=TRANSACTION_COLORS
    )
    
    fig_companies.update_traces(texttemplate='', textposition='inside')
    fig_companies.update_layout(
        showlegend=True,
        height=550,
        xaxis_tickangle=-45 if display_option == "Company Name" else 0,
        legend_title_text='Transaction Type',
        xaxis_title=x_label,
        yaxis_title='Number of Transactions'
    )
    st.plotly_chart(fig_companies, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Top 5 Sectors by Transaction Type")
    
    top_sectors_total = filtered_data.groupby('sector').size().reset_index(name='total_count')
    top_sectors_total = top_sectors_total.sort_values('total_count', ascending=False)
    top_sectors_list = top_sectors_total.head(5)['sector'].tolist()
    top_sectors_data = filtered_data[filtered_data['sector'].isin(top_sectors_list)]
    
    sector_type_data = top_sectors_data.groupby(['sector', 'transaction_type']).size().reset_index(name='count')
    
    sector_totals = sector_type_data.groupby('sector')['count'].sum().reset_index()
    sector_totals = sector_totals.sort_values('count', ascending=False)
    sector_order = sector_totals['sector'].tolist()
    
    sector_type_data['sector'] = pd.Categorical(sector_type_data['sector'], categories=sector_order, ordered=True)
    sector_type_data = sector_type_data.sort_values('sector')
    
    fig_sectors_stacked = px.bar(
        sector_type_data,
        x='sector',
        y='count',
        color='transaction_type',
        title="Top 5 Sectors by Transaction Type",
        color_discrete_map=TRANSACTION_COLORS
    )
    
    fig_sectors_stacked.update_traces(texttemplate='', textposition='inside')
    fig_sectors_stacked.update_layout(showlegend=True, height=550, xaxis_tickangle=-45, legend_title_text='Transaction Type')
    st.plotly_chart(fig_sectors_stacked, use_container_width=True)
    



    st.markdown("▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️") #better separation of the sections
    st.markdown("▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️") # because by accident an user may see as the same graph
    st.markdown("▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️▪️")

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    st.subheader("🏭 Top 5 Industries by Transaction Type")
    
    top_industries_total = filtered_data.groupby('industry').size().reset_index(name='total_count')
    top_industries_total = top_industries_total.sort_values('total_count', ascending=False)
    top_industries_list = top_industries_total.head(5)['industry'].tolist()
    top_industries_data = filtered_data[filtered_data['industry'].isin(top_industries_list)]
    
    industry_type_data = top_industries_data.groupby(['industry', 'transaction_type']).size().reset_index(name='count')
    
    industry_totals = industry_type_data.groupby('industry')['count'].sum().reset_index()
    industry_totals = industry_totals.sort_values('count', ascending=False)
    industry_order = industry_totals['industry'].tolist()
    
    industry_type_data['industry'] = pd.Categorical(industry_type_data['industry'], categories=industry_order, ordered=True)
    industry_type_data = industry_type_data.sort_values('industry')
    
    fig_industries_stacked = px.bar(
        industry_type_data,
        x='industry',
        y='count',
        color='transaction_type',
        title="Top 5 Industries by Transaction Type",
        color_discrete_map=TRANSACTION_COLORS
    )
    
    fig_industries_stacked.update_traces(texttemplate='', textposition='inside')
    fig_industries_stacked.update_layout(showlegend=True, height=550, xaxis_tickangle=-45, legend_title_text='Transaction Type')
    st.plotly_chart(fig_industries_stacked, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📌 Data Summary")
    st.markdown(f"""
    - **Filter Applied**: {filter_display}
    - **Display Option**: {display_option}
    - **Total Transactions**: {len(filtered_data):,}
    """)

# Query Analysis Page
def query_analysis():
    st.title("📋 Query Analysis")
    st.markdown("### Answers to Specific Business Questions")
    
    data = load_data()
    
    # Q1: Top 5 sectors by number of SELL transactions in China
    st.markdown("---")
    st.subheader("📊 Q1: Top 5 Sectors by SELL Transactions in China")
    
    # Filter for SELL transactions in China
    china_sell_data = data[
        (data['transaction_type'] == 'SELL') &
        (data['country_name'] == 'China')
    ]
    
    result1 = china_sell_data.groupby('sector').size().reset_index(name='transaction_count')
    result1 = result1.sort_values('transaction_count', ascending=False)
    result1_top5 = result1.head(5)
    
    st.dataframe(result1_top5, use_container_width=True)
    
    # Q1 Graph
    if len(result1_top5) > 0:
        fig_q1 = px.bar(
            result1_top5,
            x='sector',
            y='transaction_count',
            title="Top 5 Sectors - SELL Transactions in China",
            text='transaction_count',
            color_discrete_sequence=['#DC143C']
        )
        fig_q1.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_q1.update_layout(showlegend=False, height=600, xaxis_tickangle=-45)
        st.plotly_chart(fig_q1, use_container_width=True)
    
    # Q2: Top 5 industries by number of BUY transactions in Q4
    st.markdown("---")
    st.subheader("📈 Q2: Top 5 Industries by BUY Transactions in Q4")
    
    # Filter for BUY transactions in Q4
    q4_buy_data = data[
        (data['transaction_type'] == 'BUY') &
        (data['quarter'] == 'Q4')
    ]
    
    # Group by industry and count transactions
    result2 = q4_buy_data.groupby('industry').size().reset_index(name='transaction_count')
    result2 = result2.sort_values('transaction_count', ascending=False)
    result2_top5 = result2.head(5)
    
    st.dataframe(result2_top5, use_container_width=True)
    
    # Q2 Graph
    if len(result2_top5) > 0:
        fig_q2 = px.bar(
            result2_top5,
            x='industry',
            y='transaction_count',
            title="Top 5 Industries - BUY Transactions in Q4",
            text='transaction_count',
            color_discrete_sequence=['#2E8B57']
        )
        fig_q2.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_q2.update_layout(showlegend=False, height=600, xaxis_tickangle=-45)
        st.plotly_chart(fig_q2, use_container_width=True)
    
    # Q3: Rank all quarters by total number of transactions (BUY + SELL)
    st.markdown("---")
    st.subheader("📊 Q3: Quarter Ranking by BUY + SELL Transactions (Stacked)")
    
    # Filter for BUY and SELL transactions only
    buy_sell_data = data[data['transaction_type'].isin(['BUY', 'SELL'])]
    
    # Group by quarter and transaction type for stacked chart
    result3_stacked = buy_sell_data.groupby(['quarter', 'transaction_type']).size().reset_index(name='transaction_count')
    
    result3_total = buy_sell_data.groupby('quarter').size().reset_index(name='total_transaction_count')
    result3_total = result3_total.sort_values('total_transaction_count', ascending=False)
    
    st.dataframe(result3_total, use_container_width=True)
    
    # Q3 Stacked Graph
    if len(result3_stacked) > 0:
    
        result3_stacked = sort_quarters(result3_stacked)
        
        fig_q3 = px.bar(
            result3_stacked,
            x='quarter',
            y='transaction_count',
            color='transaction_type',
            title="Quarter Ranking - BUY + SELL Transactions (Stacked by Type)",
            labels={'transaction_count': 'Number of Transactions', 'quarter': 'Quarter'},
            color_discrete_map=TRANSACTION_COLORS
        )
        fig_q3.update_traces(texttemplate='', textposition='inside')
        fig_q3.update_layout(
            showlegend=True,
            height=600,
            legend_title_text='Transaction Type',
            xaxis_title='Quarter',
            yaxis_title='Number of Transactions'
        )
        st.plotly_chart(fig_q3, use_container_width=True)

# AI-Powered Visualization Page
def ai_chat_analysis():
    st.title("🔍 AI-Powered Visualization")
    st.markdown("""
    ### Generate visualizations using natural language queries
    Powered by Microsoft LIDA and OpenAI GPT-4
    """)
    
    # API Key Input -  the api will not be stored!!
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to continue")
        return
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Load the default dataset
    data = load_data()
    
    # start lida
    lida = Manager(text_gen=llm("openai"))
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
    
    # Summarize the data, necessary also for the creation fo the graph
    with st.spinner("Analyzing your data..."):
        summary = lida.summarize(data, summary_method="default", textgen_config=textgen_config)
    
    st.markdown("### Visualization Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        model = st.selectbox("Select model:", ["gpt-4", "gpt-3.5-turbo"], index=0)
    with col2:
        max_tokens = st.number_input("Max tokens:", min_value=512, max_value=4000, value=876)
    
    st.markdown("### Ask Your Question")
    user_question = st.text_area("What would you like to visualize?",
                                "What are the trends in transaction types over quarters?")
    
    if st.button("Generate Visualization"):
        with st.spinner("Creating visualization..."):
            textgen_config = TextGenerationConfig(
                n=1,
                temperature=0.2,
                model=model,
                max_tokens=max_tokens,
                use_cache=True
            )
            
            # Generate visualization
            charts = lida.visualize(
                summary=summary,
                goal=user_question,
                textgen_config=textgen_config,
                library="matplotlib"
            )
            
            # Display results
            if charts:
                st.success("Visualization generated successfully!")
                st.markdown("### Generated Visualization")
                
                # Display the image
                chart = charts[0]
                display_image(chart.raster)
                

            
            else:
                st.error("Could not generate visualization. Please try a different question.")

# Main function
def main():
    st.sidebar.title("🧭 Navigation")
    
    # Navigation options
    nav_options = ["Main Dashboard", "Query Analysis", "🔍 AI Visualization"]
    page = st.sidebar.selectbox("Choose a page:", nav_options)
    
    if page == "Main Dashboard":
        main_dashboard()
    elif page == "Query Analysis":
        query_analysis()
    elif page == "🔍 AI Visualization":
        ai_chat_analysis()

if __name__ == "__main__":
    main()