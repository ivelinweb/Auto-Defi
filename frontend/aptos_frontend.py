import streamlit as st
import requests
import pandas as pd
import datetime
import json
import plotly.io as pio

# Page configuration for a professional, wide layout
st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished, modern look
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #1e90ff;
        color: white;
        border-radius: 8px;
        padding: 0.6rem;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #4169e1;
    }
    /* Section containers */
    .section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    .metric-card {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    /* Headings */
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Title and subtitle
st.title("📈 Trading Strategy Backtester")
st.markdown("*Backtest your trading strategies with live or CSV data across multiple exchanges and advanced algorithms.*")

# Initialize session state for exchanges
if 'exchanges' not in st.session_state:
    with st.spinner("Initializing exchanges..."):
        try:
            r = requests.get("http://127.0.0.1:5000/exchanges", timeout=5)
            if r.status_code == 200:
                st.session_state.exchanges = r.json().get("exchanges", [])
            else:
                st.error("Failed to fetch exchanges. Check backend server.")
                st.session_state.exchanges = []
        except Exception as e:
            st.error("Cannot connect to backend. Ensure server is running at http://127.0.0.1:5000")
            st.session_state.exchanges = []

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Trading Strategy", "Documentation", "About"])

# TAB 1: Trading Strategy
with tab1:
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Backtest Configuration")
    st.sidebar.markdown("*Set up your trading parameters below.*")
    
    with st.sidebar.expander("🌐 Exchange & Symbol", expanded=True):
        selected_exchange = st.selectbox(
            "Exchange", 
            st.session_state.exchanges, 
            key="selected_exchange",
            help="Select the exchange to fetch data from."
        )
        if st.button("Fetch Symbols"):
            with st.spinner(f"Loading symbols for {selected_exchange}..."):
                try:
                    r2 = requests.get(f"http://127.0.0.1:5000/symbols?exchange={selected_exchange}", timeout=5)
                    if r2.status_code == 200:
                        st.session_state.symbols = r2.json().get("symbols", [])
                        st.sidebar.success(f"Loaded {len(st.session_state.symbols)} symbols.")
                    else:
                        st.sidebar.error("Failed to fetch symbols.")
                        st.session_state.symbols = []
                except Exception as e:
                    st.sidebar.error("Error fetching symbols. Verify backend server.")
                    st.session_state.symbols = []
        selected_symbol = st.selectbox(
            "Symbol", 
            st.session_state.get("symbols", []),
            key="selected_symbol",
            help="Select the trading pair to backtest."
        ) if 'symbols' in st.session_state else st.selectbox("Symbol", [], disabled=True)

    with st.sidebar.expander("👤 User Profile", expanded=True):
        username = st.text_input("Username", "default_user", help="Your identifier for this session.")

    with st.sidebar.expander("📊 RSI Settings", expanded=True):
        rsi_period = st.number_input("RSI Period", 2, 50, 14, help="Periods for RSI calculation.")
        buy_threshold = st.number_input("Buy Threshold", 1.0, 99.0, 30.0, help="RSI level to buy or cover.")
        sell_threshold = st.number_input("Sell Threshold", 1.0, 99.0, 70.0, help="RSI level to sell or short.")

    with st.sidebar.expander("📈 MA Settings", expanded=True):
        ma_period = st.number_input("MA Period", 2, 100, 20, help="Periods for moving average.")

    with st.sidebar.expander("⚙️ Trade Settings", expanded=True):
        trade_type = st.selectbox("Trade Type", ["long", "short"], help="Long or short trades.")
        strategy = st.selectbox(
            "Strategy", 
            ["RSI", "MA", "RSI_MA", "KAGE", "KITSUNE", "RYU", "SAKURA", "HIKARI", "TENSHI", "ZEN", "defAI"],
            help="Choose your trading strategy."
        )
        use_scratch_rsi = st.checkbox("Custom RSI", help="Use custom RSI instead of pandas-ta.")
        use_csv = st.checkbox("CSV Data", help="Use local CSV data instead of live feed.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("*Ensure backend server is running.*")
    submit_button = st.sidebar.button("Run Backtest", type="primary")

    # Main content area
    if submit_button:
        with st.spinner("Executing backtest..."):
            payload = {
                "exchange": selected_exchange,
                "symbol": selected_symbol if selected_symbol else "BTC/USDT",
                "username": username,
                "rsi_period": rsi_period,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "trade_type": trade_type,
                "strategy": strategy,
                "ma_period": ma_period,
                "use_scratch_rsi": use_scratch_rsi,
                "use_csv": use_csv
            }
            try:
                response = requests.post("http://127.0.0.1:5000/trade", json=payload, timeout=200)
                if response.status_code == 200:
                    result = response.json()
                    trades = result.get("trades", [])
                    historical_data = result.get("data", [])
                    plot_img = result.get("plot", "")
                    summary = result.get("summary", {})

                    # Dashboard layout
                    st.subheader(f"Backtest Results: {payload['symbol']} on {payload['exchange']}")

                    # Summary Metrics
                    with st.container():
                        st.markdown("#### Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Total Trades", summary.get("total_trades", 0))
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            winning = summary.get("winning_trades", 0)
                            total = summary.get("total_trades", 1)
                            st.metric("Win Rate", f"{winning} ({int(winning/total*100)}%)")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Total Return", f"{summary.get('total_profit_pct', 0):.2f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Avg. Trade", f"{summary.get('avg_profit_per_trade', 0):.2f}%")
                            st.markdown('</div>', unsafe_allow_html=True)

                    st.divider()

                    # Technical Analysis Plot
                    with st.container():
                        st.markdown("#### Technical Analysis")
                        if plot_img:
                            if strategy.upper() == "DEFAI":
                                fig = pio.from_json(plot_img)
                                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
                            else:
                                st.image(f"data:image/png;base64,{plot_img}", use_column_width=True)
                        else:
                            st.warning("No plot generated.")

                    st.divider()

                    # Trade Details
                    with st.container():
                        st.markdown("#### Trade Log")
                        if trades:
                            trade_df = pd.DataFrame(trades)
                            for col in ['entry_rsi', 'exit_rsi']:
                                if col not in trade_df.columns:
                                    trade_df[col] = ""
                            display_cols = ['symbol', 'trade_type', 'entry_time', 'entry_price', 'entry_rsi', 
                                            'exit_time', 'exit_price', 'exit_rsi', 'profit_pct']
                            trade_df_display = trade_df[display_cols].copy()
                            trade_df_display['profit_pct'] = trade_df_display['profit_pct'].apply(
                                lambda x: f"{x:.2f}%" if x != "" else "N/A"
                            )
                            st.dataframe(trade_df_display, use_container_width=True, height=300)
                            csv = trade_df.to_csv(index=False)
                            st.download_button(
                                label="Download Trades (CSV)",
                                data=csv,
                                file_name=f"trades_{payload['symbol'].replace('/', '')}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No trades executed. Try adjusting parameters.")

                    # Historical Data
                    with st.expander("📜 Historical Data", expanded=False):
                        if historical_data:
                            hist_df = pd.DataFrame(historical_data)
                            st.dataframe(hist_df, use_container_width=True)
                        else:
                            st.info("No historical data available.")
                else:
                    st.error(f"Backtest failed: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.info("Verify backend server is running at http://127.0.0.1:5000")
    else:
        with st.container():
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.info("👈 Configure parameters in the sidebar and click 'Run Backtest' to start.")
            st.markdown("### Strategy Overview")
            st.markdown("""
            - **RSI:** Buy when RSI crosses above buy threshold; sell below sell threshold.
            - **MA:** Trade on price crossover of moving average.
            - **RSI_MA:** Combines RSI and MA signals.
            - **Advanced (KAGE, KITSUNE, RYU, SAKURA, HIKARI, TENSHI, ZEN):** Unique indicator-based strategies.
            - **defAI:** AI-driven strategy with yield optimization and Aptos blockchain integration.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Documentation
with tab2:
    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("Documentation")
        st.markdown("### Trading Strategies")
        st.markdown("""
        - **RSI:** Momentum oscillator for overbought/oversold conditions.
        - **MA:** Simple moving average for trend detection.
        - **RSI_MA:** Hybrid strategy combining RSI and MA.
        - **KAGE:** Volatility regime detection.
        - **KITSUNE:** Price pattern recognition.
        - **RYU:** Chaos theory metrics.
        - **SAKURA:** Pivot point regression.
        - **HIKARI:** PCA-based momentum.
        - **TENSHI:** Local extrema analysis.
        - **ZEN:** Market cycle indicators.
        - **defAI:** AI agent for yield, risk, and blockchain integration.
        """)
        st.markdown("### Data Sources")
        st.markdown("- **Live:** CCXT exchange data.\n- **CSV:** Local minute-level data.")
        st.markdown("### Tech Stack")
        st.markdown("- **Backend:** Flask, CCXT, Pandas, Plotly, aptos-sdk\n- **Frontend:** Streamlit")
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: About
with tab3:
    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("About")
        st.markdown("""
        **Trading Strategy Backtester** is a proof-of-concept tool for testing trading algorithms.

        - **Features:** Multi-strategy, multi-exchange, customizable, AI-driven with blockchain.
        - **Developed by:** Trading enthusiasts leveraging Flask and Streamlit.
        - **Version:** 1.0 | © 2025
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*Trading Strategy Backtester v1.0 | Powered by Streamlit*")
