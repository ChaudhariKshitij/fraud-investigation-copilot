"""Streamlit UI. Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from copilot import cases, chat, llm
from copilot.data_gen import generate
from copilot.detector import score
from copilot.investigator import investigate_all
from copilot.report import build_report

DB = "cases.db"
st.set_page_config(page_title="Fraud Investigation Copilot", layout="wide")
st.title("Fraud Investigation Copilot")

with st.sidebar:
    st.header("0. Narrative engine")
    mode = st.radio("How should case explanations be written?",
                    ["1. Template (free, offline)", "2. Claude (enter Anthropic API key)"], key="engine_mode")
    api_key = ""  # "" = explicitly no key -> template
    if mode.startswith("2"):
        api_key = st.text_input("Anthropic API key", type="password", key="api_key_input",
                                help="Kept only in this browser session's memory. Never saved to disk or logged.")
        if not api_key:
            st.info("Enter a key to use Claude. Until then the template is used.")
        elif st.button("Test key"):
            ok, msg = llm.check_key(api_key)
            (st.success if ok else st.error)(msg)
    st.divider()
    st.header("1. Data")
    up = st.file_uploader("Upload transactions CSV", type="csv")
    st.caption("Required columns: txn_id, customer_id, timestamp, amount, country, merchant, device_id")
    if st.button("Use demo data"):
        st.session_state["raw"] = generate()
    if up is not None:
        st.session_state["raw"] = pd.read_csv(up)
    contamination = st.slider("Share flagged as anomalous", 0.005, 0.05, 0.02, 0.005)

st.caption(f"Narrative backend: **{llm.backend_name(api_key)}** "
           "(choose the engine in the sidebar)")

raw = st.session_state.get("raw")
if raw is None:
    st.info("Upload a CSV or click **Use demo data** in the sidebar.")
    st.stop()

missing = {"txn_id", "customer_id", "timestamp", "amount", "country", "merchant", "device_id"} - set(raw.columns)
if missing:
    st.error(f"Missing columns: {sorted(missing)}")
    st.stop()


@st.cache_data(show_spinner="Scoring transactions…")
def run_detection(df, c):
    return score(df, contamination=c)


df = run_detection(raw, contamination)
tab1, tab2, tab3, tab4 = st.tabs(["Anomalies", "Investigate", "Chat with data", "Case dashboard"])

with tab1:
    fl = df[df.flagged == 1]
    a, b, c = st.columns(3)
    a.metric("Transactions", f"{len(df):,}")
    b.metric("Flagged", f"{len(fl):,}")
    c.metric("Flagged value", f"{fl.amount.sum():,.0f}")
    st.dataframe(fl.sort_values("anomaly_score", ascending=False)[
        ["txn_id", "customer_id", "timestamp", "amount", "country", "merchant", "anomaly_score"]],
        width='stretch', hide_index=True)

with tab2:
    n = st.number_input("Customers to investigate (highest score first)", 1, 100, 10)
    if st.button("Run investigation agent"):
        with st.spinner("Investigating…"):
            st.session_state["inv"] = investigate_all(df, top_n=int(n), api_key=api_key)
    for ev in st.session_state.get("inv", []):
        with st.expander(f"{ev['customer_id']} · {ev['risk']} · {ev['typology']}"):
            st.write(ev["explanation"])
            for f in ev["red_flags"]:
                st.markdown(f"- {f}")
            c1, c2 = st.columns(2)
            if c1.button("Open case", key=f"open{ev['customer_id']}"):
                cases.open_case(DB, ev)
                st.success("Case opened")
            c2.download_button("Download report (.md)", build_report(ev), f"{ev['customer_id']}_report.md",
                               key=f"dl{ev['customer_id']}")

with tab3:
    st.caption("Try: “how many are flagged?”, “top customers”, “flagged by country”, “C0012”, “what is structuring?”")
    if "msgs" not in st.session_state:
        st.session_state["msgs"] = []
    for role, text in st.session_state["msgs"]:
        st.chat_message(role).markdown(text)
    if q := st.chat_input("Ask about the transactions"):
        st.session_state["msgs"] += [("user", q), ("assistant", chat.answer(df, q))]
        st.rerun()

with tab4:
    rows = cases.list_cases(DB)
    if not rows:
        st.info("No cases yet. Open one from the Investigate tab.")
    else:
        st.dataframe(pd.DataFrame(rows)[["id", "customer_id", "status", "risk", "typology", "updated_at"]],
                     width='stretch', hide_index=True)
        cid = st.selectbox("Case", [r["id"] for r in rows])
        cur = next(r for r in rows if r["id"] == cid)
        status = st.selectbox("Status", cases.STATUSES, index=cases.STATUSES.index(cur["status"]))
        notes = st.text_area("Notes", cur["notes"])
        if st.button("Save"):
            cases.update_case(DB, cid, status=status, notes=notes)
            st.rerun()
