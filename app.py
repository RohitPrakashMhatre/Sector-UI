import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
# import process_functions as func  # <- plug in later if needed

# --------------------------
# Config / Inputs
# --------------------------
st.set_page_config(page_title="Stocks & Sectors Analysis", layout='wide', initial_sidebar_state='expanded')
st.title("📈 Stocks & Sectors Analysis")

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "excels", "sectors_stocks.csv")
#file_path = "C://Users//Rohit Mhatre//Documents//SectorSystem//excels//sectors_stocks.csv"
symbols_data = pd.read_csv(file_path)

# Choose which column in your CSV is the ticker. Adjust if needed:
TICKER_COL = 'SYMBOL' if 'SYMBOL' in symbols_data.columns else ('NSE CODE' if 'NSE CODE' in symbols_data.columns else None)
if TICKER_COL is None:
    st.stop()  # fail fast with message
symbols_data[TICKER_COL] = symbols_data[TICKER_COL].astype(str).str.replace(':', '.')  # normalize
df = symbols_data.head(200).copy().set_index(TICKER_COL)

# Dates & week labels (keep as you provided)
dates = ["2024-12-30","2025-02-05","2025-01-27","2025-01-02","2024-12-20","2024-12-05","2024-11-21","2024-11-07","2024-12-17","2024-12-24"]
weeks = ['base','Week 1','Week 2','Week 3','Week 4','Week 5','Week 6','Week 7','Week 8','Week 9']

# Convert to pd.Timestamp once
date_index = [pd.Timestamp(d) for d in dates]

# Pre-create columns so assignment is safe
for col in ['Current'] + weeks:
    if col not in df.columns:
        df[col] = np.nan


def chunk_batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def safe_fetch(symbol: str, period: str = '1y', attempts: int = 3) -> pd.DataFrame:
    for attempt in range(attempts):
        try:
            hist = yf.Ticker(symbol).history(period=period, auto_adjust=False)
            if not hist.empty:
                hist.index = pd.to_datetime(hist.index).tz_localize(None)
                return hist
        except Exception as e:
            if attempt == attempts - 1:
                st.warning(f"Failed {symbol} after {attempts} tries: {e}")
            else:
                time.sleep(0.8 + 0.3 * attempt)
    return pd.DataFrame()

def close_on_or_before(history: pd.DataFrame, target: pd.Timestamp):
    """Close price for last trading session on/before target date."""
    if history.empty:
        return None
    idx = history.index
    # exact match
    if target in idx:
        return float(history.at[target, 'Close'])
    # previous available date
    prev = idx[idx <= target]
    if len(prev) == 0:
        return None
    return float(history.loc[prev.max(), 'Close'])

def fetch_batch(batch, week_labels, target_dates, period='1y'):
    out = {}
    for symbol in batch:
        hist = safe_fetch(symbol, period=period)
        if hist.empty:
            continue
        sym_map = {}
        sym_map['Current'] = float(np.ceil(hist['Close'].iloc[-1] * 100) / 100)
        for wlabel, tdate in zip(week_labels, target_dates):
            px = close_on_or_before(hist, tdate)
            if px is not None:
                sym_map[wlabel] = float(np.ceil(px * 100) / 100)
        if sym_map:
            out[symbol] = sym_map
    return out

def run_multithreaded(symbols, week_labels, target_dates, period='1y', max_workers=10, batch_size=25):
    batches = list(chunk_batch(symbols, batch_size))
    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(fetch_batch, b, week_labels, target_dates, period) for b in batches]
        progress = st.progress(0.0, text="Fetching...")
        done = 0

        total = len(futs)
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            progress.progress(done / total, text=f"Fetching... ({done}/{total})")
    duration = time.time() - start
    return results, duration

def merge_results(df_in: pd.DataFrame, fetched_list):
    df_out = df_in.copy()
    for blob in fetched_list:
        for sym, cols in blob.items():
            if sym in df_out.index:
                for col, val in cols.items():
                    if col not in df_out.columns:
                        df_out[col] = np.nan
                    df_out.at[sym, col] = val
    return df_out

# --------------------------
# Sidebar controls
# --------------------------


fetch_tab, tab_2, tab_3, tab_4, tab_5 = st.tabs(
    ["Fetch","Top Sectors","Histogram", "Volatility", "Sectors Ranking"]
)


with fetch_tab:
    st.subheader("Fetch data here")

    c1, c2, c3 = st.columns(3)

    period = c1.selectbox("yfinance period", ["6mo","1y","2y","5y"], index=1)
    batch_size = c2.slider("Batch size", 5, 100, 25, step=5)
    max_workers = c3.slider("Max workers (threads)", 1, 32, 10)

    with st.expander("Preview tickers"):
        st.dataframe(df.head(20))

    # Session state
    if "result_df" not in st.session_state:
        st.session_state["result_df"] = None
    if "runtime" not in st.session_state:
        st.session_state["runtime"] = None
    if st.button("🚀 Fetch data now"):
        symbols = df.index.tolist()
        fetched, secs = run_multithreaded(
            symbols=symbols,
            week_labels=weeks,
            target_dates=date_index,
            period=period,
            max_workers=max_workers,
            batch_size=batch_size,
        )
        merged = merge_results(df, fetched)
        st.session_state["result_df"] = merged
        st.session_state["runtime"] = secs
        st.success(f"Completed in {secs:.1f}s — added/updated columns: "
                f"{sum(c in merged.columns for c in (['Current'] + weeks))}")

    # --------------------------
    # Results + Download
    # --------------------------
    if st.session_state["result_df"] is not None:
        st.subheader("Results")
        st.dataframe(st.session_state["result_df"].sort_index())

        out_csv = st.session_state["result_df"].reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=out_csv,
            file_name=f"sector_prices_{dt.date.today().isoformat()}.csv",
            mime="text/csv"
        )

def _tz_naive_sorted(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return hist
    h = hist.copy()
    h.index = pd.to_datetime(h.index).tz_localize(None)
    return h.sort_index()

def close_on_or_before(hist: pd.DataFrame, target: pd.Timestamp):
    """Close for the last trading bar on/before target; None if all bars are after target."""
    if hist.empty:
        return None
    h = _tz_naive_sorted(hist)
    idx = h.index
    pos = idx.searchsorted(pd.Timestamp(target), side="right") - 1
    if pos < 0:
        return None
    return float(h["Close"].iloc[pos])

def pre_processing(df: pd.DataFrame, dates, weeks):
    """
    - expects weeks like ['base','Week 1',...,'Week 9'] and matching dates list
    - computes Live % columns relative to 'base'
    - fetches NIFTY (^NSEI) and adds its Live deltas to each stock's Live columns
    """
    df = df.copy()

    # Keep only rows having 'base' if it exists; otherwise skip the drop
    if 'base' in df.columns:
        df.dropna(subset=['base'], inplace=True)

    # --- Stock live % (only for columns that actually exist) ---
    if 'Current' in df.columns and 'base' in df.columns:
        df['Live'] = np.ceil(((df['Current'] - df['base']) / df['base']) * 100)

    for i in range(1, 10):
        col_w = f'Week {i}'
        col_l = f'Live {i}'
        if col_w in df.columns and 'base' in df.columns:
            df[col_l] = np.ceil(((df[col_w] - df['base']) / df['base']) * 100)

    # If your dataset has 'Sector'/'NSE CODE' keep them; otherwise just keep what exists
    keep_cols = ['Sector', 'NSE CODE', 'Live'] + [f'Live {i}' for i in range(1, 10)]
    keep_cols_present = [c for c in keep_cols if c in df.columns]
    if keep_cols_present:
        df = df[keep_cols_present].copy()

    # --- NIFTY (^NSEI) alignment ---
    Symbol = '^NSEI'
    hist = yf.Ticker(Symbol).history(period='2y', auto_adjust=False)
    hist = _tz_naive_sorted(hist)

    # map date → label using "last bar on/before date"
    date_ts = [pd.Timestamp(d) for d in dates]
    label_by_date = dict(zip(date_ts, weeks))  # e.g. {2024-12-30:'base', 2025-02-05:'Week 1', ...}

    nifty_map = {lbl: np.nan for lbl in weeks}
    for tdate, lbl in label_by_date.items():
        px = close_on_or_before(hist, tdate)
        if px is not None:
            nifty_map[lbl] = float(np.ceil(px * 100) / 100)

    # current = last bar
    nifty_current = float(np.ceil(hist['Close'].iloc[-1] * 100) / 100) if not hist.empty else np.nan

    # base must exist to compute %s
    if not np.isnan(nifty_map.get('base', np.nan)) and not np.isnan(nifty_current):
        nifty_live = {}
        # Live (current vs base)
        nifty_live['Live'] = round(((nifty_current - nifty_map['base']) / nifty_map['base']) * 100, 2)
        # Live i (week i vs base)
        for i in range(1, 10):
            wk_lbl = f'Week {i}'
            if not np.isnan(nifty_map.get(wk_lbl, np.nan)):
                nifty_live[f'Live {i}'] = round(((nifty_map[wk_lbl] - nifty_map['base']) / nifty_map['base']) * 100, 2)

        # add NIFTY effect to each stock Live column that exists
        for col, val in nifty_live.items():
            if col in df.columns:
                df[col] = df[col].astype(float).add(val, fill_value=0.0)

    return df

def find_top_n_live(df, n):
    df = df.copy()
    group_sec = df.groupby('Sector')[['Live','Live 1','Live 2','Live 3','Live 4','Live 5','Live 6','Live 7','Live 8','Live 9']].mean()
    group_sec = group_sec.sort_values(by='Live',ascending=False)
    group_sec.reset_index(inplace=True)
    group_sec.index = group_sec.index + 1
    top_live = group_sec.head(n)

    return top_live, group_sec

# df = pre_processing(df=df, dates=dates, weeks=weeks)
# print("Data: ", df)

with tab_2:
    st.subheader("Top Sectors")
    if "result_df" not in st.session_state or st.session_state["result_df"] is None or st.session_state["result_df"].empty:
        st.warning("No data to analyze. Go to the Fetch tab and run the fetch first.")
    else:
        base_df = st.session_state["result_df"].copy()
        try:
            df_proc = pre_processing(df=base_df, weeks=weeks, dates=dates)
        except Exception as e:
            st.error(f"pre-processing failed: {e}")
            df_proc = None

        
        top_n = st.number_input(
            "How many top sector?",
            min_value=5,
            max_value=100,
            step=1,
            value=10,
            key="top_n_input"
        )
        if df_proc.empty:
            st.warning("No data to Analyze. Please fetch data and try again")
        else:
            top_live, group_sec = find_top_n_live(df_proc, int(top_n))
            if top_live.empty:
                st.warning("No data to Analyze. Please fetch data if not else try again.")
            else:
                live_cols = ["Live"] + [c for c in top_live.columns if c.startswith("Live ") and c.split()[-1].isdigit()]
                weekly_df = top_live.set_index("Sector")[live_cols]

                weekly_df = weekly_df.sort_values(by="Live", ascending=False)
                st.caption("Top Sectors Table")

                st.dataframe(top_live.reset_index(drop=True), use_container_width=True)

                # Heatmap
                fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(weekly_df))))
                sns.heatmap(
                    weekly_df,
                    cmap="RdYlGn",
                    annot=True,
                    fmt=".0f",
                    cbar=False,
                    linewidths=0.5,
                    ax=ax
                )
                ax.set_title("Heatmap of Sector performance (%)")
                ax.set_xlabel("Live Weeks")
                ax.set_ylabel("Sectors")

                st.pyplot(fig)
                if "process_data" not in st.session_state:
                    st.session_state["process_data"] = df_proc
                if "weekly_df" not in st.session_state:
                    st.session_state["weekly_df"] = weekly_df
                if "top_live" not in st.session_state:
                    st.session_state["top_live"] = top_live
                if "group_sec" not in st.session_state:
                    st.session_state["group_sec"] = group_sec


with tab_3:
    st.subheader("Top sectors performance")

    hist_n = st.number_input(
        "How many top sectors?",
        min_value=5,
        max_value=300,
        value=30,
        step=5,
        key="hist_n_input"
    )

    # --- Validate session data ---
    if "group_sec" not in st.session_state or "top_live" not in st.session_state:
        st.warning("No data to analyze. Please run the previous tab (Top Sectors) first.")
    else:
        group_sec_df = st.session_state["group_sec"]
        top_live_df = st.session_state["top_live"]

        if group_sec_df is None or group_sec_df.empty:
            st.warning("Group sector data is empty.")
        else:
            # pick top N
            top_group_sec = group_sec_df.head(int(hist_n)).copy()
            top_group_sec["Live"] = top_group_sec["Live"].round(2)


            # --- Create the bar chart ---
            fig = px.bar(
                top_group_sec,
                x="Sector",
                y="Live",
                color="Sector",
                text="Live",
                title="Top Sector Performance",
                width=1200,
                height=600
            )

            fig.update_layout(
                xaxis_title="Sectors",
                yaxis_title="Live (%)",
                xaxis_tickangle=90,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)


with tab_4:
    st.subheader("Sectors Volatility")
    
    # Check if the session data exists and isn't empty
    if "weekly_df" not in st.session_state or st.session_state["weekly_df"].empty:
        st.warning("No data to analyze. Please complete the top sectors analysis first.")
    else:
        # Retrieve the dataframe from session_state
        weekly_df_data = st.session_state["weekly_df"]

        # Create a boxplot of sector volatility (transposed data for sectors in rows)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=weekly_df_data.T)  # Transpose data so each sector is a boxplot
        plt.xticks(rotation=90)
        plt.title('Sector Volatility Analysis')

        # Render the plot in Streamlit
        st.pyplot(plt)


# def improved_sectors(grid_df):
#     def find_improved_sector(grid_df):
#         condition = (grid_df['Live 1'] - grid_df['Live'] >= 2) | (grid_df['Live 2'] - grid_df['Live'] >= 2)
#         return data.loc[condition,['Sector','Live','Live 1','Live 2']]

#     improved = find_improved_sector(grid_df)
#     improved_sec = improved['Sector']
#     return improved_sec

# def find_improved_sectors_stocks(unprocessed_df, avg_df, improved_sec):

#     results = {sec: [] for sec in improved_sec}
#     dummy_df = unprocessed_df.copy().reset_index()
#     avg_sec = avg_df.groupby('Sector')['Live'].mean()

#     # print("Improved Sectors Stocks:")
#     # print("---------------------------------------------------------")

#     for imp in improved_sec:
#         if imp in avg_sec.index:
#             avg_value = avg_sec[imp]
#             # filter stocks in improved sector above average live
#             stock_list = dummy_df[
#                 (dummy_df['Sector'] == imp) & 
#                 (dummy_df['Live'] >= avg_value)
#             ]['NSE CODE'].to_list()

#             results[imp] = stock_list

#             # if stock_list: 
#             #     print(f"{imp}: {stock_list}")

#     return results

def get_improved_sectors(df: pd.DataFrame) -> list[str]:
    """Sectors where Week1 or Week2 is at least 2% above base 'Live'."""
    need = ["Sector", "Live", "Live 1", "Live 2"]
    if any(c not in df.columns for c in need):
        return []
    tmp = df.copy()
    for c in ["Live", "Live 1", "Live 2"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp["Live 1 - Live"] = tmp["Live 1"] - tmp["Live"]
    tmp["Live 2 - Live"] = tmp["Live 2"] - tmp["Live"]
    
    # Check the differences
    condition = ((tmp["Live 1 - Live"] >= 2) | (tmp["Live 2 - Live"] >= 2))
    
    # Output which sectors meet the condition
    # st.write(tmp[["Sector", "Live", "Live 1", "Live 2", "Live 1 - Live", "Live 2 - Live"]])  # Debugging output
    
    return tmp.loc[condition, "Sector"].dropna().unique().tolist()


def map_improved_stocks(unprocessed_df: pd.DataFrame,
                        improved_secs: list[str]) -> dict[str, list[str]]:
    """For each improved sector, list stocks with Live >= sector's average Live."""
    if unprocessed_df is None or unprocessed_df.empty or not improved_secs:
        return {}

    df = unprocessed_df.copy()
    sec_col = "Sector" if "Sector" in df.columns else ("Sectors" if "Sectors" in df.columns else None)
    code_col = "NSE CODE" if "NSE CODE" in df.columns else ("SYMBOL" if "SYMBOL" in df.columns else None)
    if not sec_col or not code_col or "Live" not in df.columns:
        return {}

    df["Live"] = pd.to_numeric(df["Live"], errors="coerce")
    df = df.dropna(subset=["Live"])
    avg_live = df.groupby(sec_col)["Live"].mean()

    out: dict[str, list[str]] = {}
    for sec in improved_secs:
        if sec in avg_live.index:
            thr = avg_live.loc[sec]
            stocks = (
                df[(df[sec_col] == sec) & (df["Live"] >= thr)][code_col]
                .dropna().astype(str).tolist()
            )
            out[sec] = stocks
    return out

with tab_5:
    
    rank_tab, stocks_tab = st.tabs(["Rank Sectors", "Top Stocks"])
    with rank_tab:
        st.subheader("Rank Analysis")

        if "top_live" not in st.session_state or st.session_state["top_live"] is None or st.session_state["top_live"].empty:
            st.warning("Perform top sectors analysis first (Tab 2).")
        else:
            top_live_data = st.session_state["top_live"].copy()

            # Rank only Live columns that exist
            rank_cols_all = ['Live'] + [f'Live {i}' for i in range(1, 10)]
            rank_cols = [c for c in rank_cols_all if c in top_live_data.columns]
            grid_df = top_live_data.copy()
            for col in rank_cols:
                grid_df[col] = pd.to_numeric(grid_df[col], errors="coerce")
                grid_df[col] = grid_df[col].rank(method='dense', ascending=False).astype('Int64')

            # Heat-colored grid
            styled_df = grid_df.style
            for col in rank_cols:
                styled_df = styled_df.background_gradient(cmap='RdYlGn_r', subset=[col])
            st.dataframe(styled_df, use_container_width=True)
            st.session_state["grid_df"] = grid_df  # stash ranked grid

            # ---------- Improved sectors & stocks (shown RIGHT HERE) ----------
            st.markdown("### 📈 Improved Stocks (in this grid tab)")
            improved_secs = get_improved_sectors(grid_df)  # use raw values, not ranks
            st.session_state["improved_sectors"] = improved_secs

            base_rows = st.session_state.get("process_data")  # raw row-level DF set earlier (after fetch)
            if base_rows is None or (isinstance(base_rows, pd.DataFrame) and base_rows.empty):
                st.info("Raw row-level data not available (process_data). Skipping improved stocks list.")
            else:
                improved_map = map_improved_stocks(base_rows, improved_secs)
                st.session_state["improved_stocks"] = improved_map

                # Summary table (sector → count + preview of tickers)
                rows = []
                for sec in improved_secs:
                    tickers = improved_map.get(sec, [])
                    rows.append({
                        "Sector": sec,
                        "Improved Stock Count": len(tickers),
                        "Stocks (preview)": "".join(tickers[:30])
                    })
                if rows:
                    summary_df = pd.DataFrame(rows).sort_values("Improved Stock Count", ascending=False)
                    st.dataframe(summary_df, use_container_width=True)

                    # Expanders with all tickers per sector
                    for sec in improved_secs:
                        tickers = improved_map.get(sec, [])
                        with st.expander(f"{sec} — {len(tickers)} stocks"):
                            if not tickers:
                                st.write("None")
                            else:
                                cols = st.columns(6)
                                for i, t in enumerate(tickers):
                                    cols[i % 6].markdown(f"`{t}`")

                    # Download CSV
                    flat = [(sec, t) for sec in improved_secs for t in improved_map.get(sec, [])]
                    if flat:
                        st.download_button(
                            "⬇️ Download improved stocks (CSV)",
                            data=pd.DataFrame(flat, columns=["Sector", "Stock"]).to_csv(index=False).encode("utf-8"),
                            file_name="improved_stocks.csv",
                            mime="text/csv",
                        )
                else:
                    st.info("No sectors met the improvement rule (Week 1 or 2 ≥ 2% above base).")


