import logging
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from risklens.core import FEATURES, analyze, generate_transactions
from risklens.db import (
    get_recent_analyses,
    get_scored_for_analysis,
    init_db,
    record_review,
    save_analysis,
)

# Structured logging for this page.
log_path = os.getenv("RISKLENS_LOG_PATH", "risklens.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="a"),
    ],
)
logger = logging.getLogger("risklens")

# A demo login for this Streamlit page. The password comes from the environment. This is not the scoring API.
DEMO_USERS = {
    "analyst": os.getenv("RISKLENS_PASSWORD", "demo"),  # override via env in prod
}

def check_auth() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
    return st.session_state.authenticated

def login_form():
    st.title("RiskLens Login")
    st.caption("Phase 1 hardening — basic auth (demo credentials)")
    with st.form("login"):
        username = st.text_input("Username", value="analyst")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
        if submitted:
            if username in DEMO_USERS and DEMO_USERS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                logger.info("User %s authenticated", username)
                st.rerun()
            else:
                st.error("Invalid credentials")
                logger.warning("Failed login attempt for user %s", username)

# The original console. It flags synthetic rows for a person to review. A flag is not a fraud verdict.
def main():
    init_db()
    st.set_page_config(page_title="RiskLens | Investigation Console", page_icon="🔎", layout="wide")

    if not check_auth():
        login_form()
        return

    st.title("RiskLens")
    st.caption("Synthetic transaction anomaly detection · Research demonstration · Not a fraud verdict")
    st.sidebar.success(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Log out"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

    with st.sidebar:
        st.header("Detection settings")
        n = st.slider("Synthetic transactions", 200, 5000, 1200, 100)
        contamination = st.slider("Expected anomaly proportion", 0.01, 0.20, 0.05, 0.01)
        seed = st.number_input("Random seed", 0, 100000, 42)
        uploaded = st.file_uploader("Or upload a CSV with required feature columns", type=["csv"])
        persist = st.checkbox("Persist this run to local DB", value=True)

    try:
        logger.info("Starting analysis | n=%s contamination=%s seed=%s uploaded=%s", n, contamination, seed, uploaded is not None)
        df = pd.read_csv(uploaded) if uploaded else generate_transactions(n, int(seed))
        missing = set(FEATURES) - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {sorted(missing)}")
            logger.error("Missing columns: %s", missing)
            st.stop()
        for col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors="raise")
        if "transaction_id" not in df:
            df["transaction_id"] = [f"TX-{i:06d}" for i in range(len(df))]
        if "customer_id" not in df:
            df["customer_id"] = "Unknown"
        result = analyze(df, contamination)
        logger.info("Analysis complete | rows=%s flagged=%s", len(result), result.flagged.sum())
    except (ValueError, KeyError, TypeError) as e:
        st.error(f"Unable to process transactions: {e}")
        logger.exception("Analysis failed")
        st.stop()

    analysis_id = None
    if persist:
        try:
            notes = f"user={st.session_state.username}"
            analysis_id = save_analysis(int(seed), float(contamination), result, notes)
            st.success(f"Run persisted (analysis_id={analysis_id})")
            logger.info("Persisted analysis_id=%s", analysis_id)
        except Exception as e:
            st.warning(f"Could not persist: {e}")
            logger.exception("Persistence failed")

    flagged = result[result.flagged]
    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{len(result):,}")
    b.metric("Flagged for review", f"{len(flagged):,}")
    c.metric("Review rate", f"{len(flagged)/len(result):.1%}")
    d.metric("Flagged value", f"£{flagged.amount.sum():,.2f}")

    st.subheader("Transaction risk landscape")
    fig = px.scatter(
        result,
        x="amount",
        y="anomaly_score",
        color="flagged",
        hover_data=["transaction_id", "review_reasons"],
        log_x=True,
        title="Higher scores indicate greater model anomaly",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Investigation queue")
    only_flagged = st.checkbox("Show flagged transactions only", value=True)
    view = flagged if only_flagged else result
    st.dataframe(
        view[["transaction_id", "customer_id", "amount", "anomaly_score", "flagged", "review_reasons"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Export investigation queue CSV",
        view.to_csv(index=False),
        "risklens_investigations.csv",
        "text/csv",
    )

    if not view.empty and analysis_id:
        st.subheader("Record human review outcome")
        # Map transaction_id back to scored_transaction row id for the latest analysis
        scored_rows = get_scored_for_analysis(analysis_id)
        id_map = {r["transaction_id"]: r["id"] for r in scored_rows}
        selected_tid = st.selectbox("Select transaction to review", view.transaction_id.tolist())
        scored_id = id_map.get(selected_tid)
        if scored_id:
            col1, col2 = st.columns(2)
            with col1:
                decision = st.selectbox(
                    "Decision",
                    ["approved", "rejected", "escalated", "needs_more_info"],
                )
            with col2:
                notes = st.text_area("Notes", height=100)
            if st.button("Save review decision"):
                try:
                    record_review(
                        scored_id,
                        decision,
                        notes,
                        reviewer=st.session_state.username,
                    )
                    st.success(f"Review saved for {selected_tid}")
                    logger.info("Review recorded | tx=%s decision=%s by=%s", selected_tid, decision, st.session_state.username)
                except Exception as e:
                    st.error(f"Failed to save review: {e}")
                    logger.exception("Review save failed")
        else:
            st.warning("Could not map transaction to persisted row.")

        # Show details
        record = view[view.transaction_id == selected_tid].iloc[0]
        st.write(
            {
                "transaction_id": record.transaction_id,
                "customer_id": record.customer_id,
                "amount_gbp": float(record.amount),
                "anomaly_score": float(record.anomaly_score),
                "review_reasons": record.review_reasons,
            }
        )
        st.info("Analyst decision: manual review required. The model does not establish fraud or wrongdoing.")

    # Recent runs sidebar section
    with st.sidebar:
        st.header("Recent analyses")
        recent = get_recent_analyses(5)
        if recent:
            for r in recent:
                st.caption(f"{r['run_at'][:19]} · id={r['id']} · flagged={r['n_flagged']}")
        else:
            st.caption("No prior runs")

    st.caption(
        "Synthetic scenario labels, when available, are not used to train the model. "
        "Review reasons are heuristic context, not causal explanations of Isolation Forest. "
        "Phase 1: basic auth + SQLite persistence + logging."
    )

if __name__ == "__main__":
    main()
