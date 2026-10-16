"""Streamlit investigator view over saved, matched development artifacts."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from api.model import EXPECTED_MODEL_SHA256, FEATURE_COLUMNS, MODEL_PATH
from dashboard.evidence import (
    CLUSTERS, CONFIG, GRAPH, PREDICTIONS, SHAP_PATH, VALIDATION,
    deterministic_summary, file_signature, load_clusters, load_evidence_table, make_evidence_record,
)
from dashboard.shap_evidence import checked_shap, load_shap_export

st.set_page_config(page_title="SentinelGraph | Investigation", page_icon="🛡️", layout="wide")
st.title("SentinelGraph AI")
st.caption("Read-only investigation workspace · saved development evidence · human review")


@st.cache_data(show_spinner="Verifying saved transaction evidence…")
def get_data(signature):
    return load_evidence_table()


@st.cache_data(show_spinner=False)
def get_shap(signature):
    return load_shap_export()


@st.cache_data(show_spinner=False)
def get_clusters(signature):
    return load_clusters()


view = st.sidebar.radio("View", ["Transaction evidence", "Retrospective clusters", "Monitoring"], key="view")
st.sidebar.caption("Anonymized proxies do not establish real-world identities. Scores do not assign fraud decisions.")

if view == "Transaction evidence":
    try:
        table = get_data(file_signature([PREDICTIONS, CONFIG, VALIDATION, GRAPH, MODEL_PATH]))
    except (OSError, ValueError, KeyError, AssertionError) as error:
        st.error("Evidence artifacts could not be loaded or did not pass verification.")
        st.caption(str(error))
        st.stop()
    st.subheader("Saved transactions")
    st.caption(f"{len(table):,} matched Session 20 evaluation records, ranked by uncalibrated score; TransactionID breaks ties.")
    count = st.select_slider("Rows shown", options=[value for value in [10, 25, 50, 100, 250, 500] if value <= len(table)], value=100)
    st.dataframe(table[["TransactionID", "TransactionAmt", "ProductCD", "model_score"]].head(count), hide_index=True,
                 column_config={"model_score": st.column_config.NumberColumn("Uncalibrated score", format="%.6f"),
                                "TransactionAmt": st.column_config.NumberColumn("Observed amount", format="%.2f")}, width="stretch")
    selected_id = st.selectbox("Transaction", table.TransactionID.astype(int).tolist(),
                               format_func=lambda value: f"Transaction {value}", key="selected_transaction")
    evidence = make_evidence_record(table, selected_id)
    st.subheader(f"Transaction {selected_id}")
    score, amount, decision = st.columns(3)
    score.metric("Uncalibrated score", f"{evidence['model_score']:.6f}")
    amount.metric("Observed amount", "Unavailable" if evidence["transaction_amount"] is None else f"{evidence['transaction_amount']:,.2f}")
    decision.metric("Decision", "Not assigned")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Historical feature snapshot")
        st.caption("Counts saved from strictly earlier observed events, not recomputed with later history.")
        st.dataframe(pd.DataFrame([{"Recorded feature": key, "Value": value} for key, value in evidence["historical_features"].items()]),
                     hide_index=True, width="stretch")
    with right:
        st.markdown("#### Grounded summary")
        st.info(deterministic_summary(evidence))
        st.caption("Deterministic evidence summary. No language model or external service is used.")
    st.markdown("#### Model attribution")
    try:
        payload, lookup = get_shap(file_signature([SHAP_PATH]))
        explanation = checked_shap(evidence, payload, lookup)
        if explanation is None:
            st.info("No verified SHAP explanation is available for this transaction.")
        else:
            contributions = pd.DataFrame([{"Feature": key, "SHAP contribution": value}
                                           for key, value in explanation["feature_contributions"].items()])
            contributions = contributions.sort_values("SHAP contribution")
            st.bar_chart(contributions.set_index("Feature"), horizontal=True)
            st.caption(f"Raw log-odds: baseline {explanation['base_value']:.6f} + contributions = {explanation['raw_output']:.6f}. "
                       "Positive contributions increase raw model output. Attributions describe this model, not causes or intent.")
    except (OSError, ValueError, KeyError, AssertionError) as error:
        st.warning("A matching, verified SHAP explanation is unavailable. The selected saved evidence remains visible.")
        st.caption(str(error))
    with st.expander("Evidence record and audit sources"):
        st.json(evidence)
    st.download_button("Download evidence JSON", json.dumps(evidence, indent=2, allow_nan=False),
                        file_name=f"transaction_{selected_id}_evidence.json", mime="application/json")

elif view == "Retrospective clusters":
    st.subheader("Retrospective proxy-linked research")
    st.info("Session 16 used a completed training sample. These components are not inference-time evidence and are not joined to Session 20 transactions.")
    try:
        components, members, edges, summary = get_clusters(file_signature(
            [CLUSTERS / name for name in ["summary.json", "component_summary.csv", "component_memberships.csv", "edge_evidence.csv"]]))
    except (OSError, ValueError, KeyError, AssertionError) as error:
        st.warning("Verified retrospective component artifacts are unavailable.")
        st.caption(str(error))
        st.stop()
    a, b, c = st.columns(3)
    a.metric("Components", len(components))
    b.metric("Covered transactions", len(members))
    c.metric("Original sample", f"{summary['sample_rows']:,}")
    show_outcomes = st.checkbox("Show retrospective outcome statistics", value=False, key="retrospective_outcomes")
    visible = components if show_outcomes else components.drop(columns=["fraud_count", "fraud_rate", "fraud_enrichment"])
    if show_outcomes:
        st.caption("Retrospective dataset labels aggregated after graph construction; these outcomes were not available as scoring-time evidence.")
    st.dataframe(visible, hide_index=True, width="stretch")
    component_id = st.selectbox("Component", components.component_id.tolist(), key="selected_component")
    selected = members.loc[members.component_id.eq(component_id)]
    selected_edges = edges.loc[edges.component_id.eq(component_id)]
    dot = ["graph proxies {", 'graph [rankdir="LR"];']
    for row in selected_edges.itertuples():
        card, address = f"card:{row.card_key}", f"address:{row.addr_key}"
        dot.extend([f"{json.dumps(card)} [label={json.dumps('card proxy ' + row.card_key)}, shape=ellipse];",
                    f"{json.dumps(address)} [label={json.dumps('address proxy ' + row.addr_key)}, shape=box];",
                    f"{json.dumps(card)} -- {json.dumps(address)};"])
    dot.append("}")
    st.graphviz_chart("\n".join(dot))
    st.caption("Edges are observed proxy co-occurrences in this retrospective sample; connectivity does not establish coordination or fraud.")
    st.dataframe(selected[["TransactionID", "TransactionDT", "card_key", "addr_key"]], hide_index=True, width="stretch")

else:
    st.subheader("Offline drift diagnostics")
    directory = ROOT / "outputs/monitoring/session23"
    try:
        drift = pd.read_csv(directory / "feature_drift.csv")
        scores = json.loads((directory / "prediction_drift.json").read_text())
        summary = json.loads((directory / "run_summary.json").read_text())
        if (len(drift) != 5 or set(drift.feature) != set(FEATURE_COLUMNS)
                or scores["model_sha256"] != EXPECTED_MODEL_SHA256 or summary["test_access"] is not False):
            raise ValueError("Monitoring scope or model differs")
        st.metric("Score distribution PSI", f"{scores['psi']:.6f}")
        st.dataframe(drift[["feature", "psi", "reference_missing_rate", "current_missing_rate"]], hide_index=True, width="stretch")
        st.caption("Last 10,000 training vs first 10,000 validation observations. Training scores are in-sample. "
                   "Historical counters accumulate over time; PSI does not prove an AP decline or a feature-pipeline defect.")
    except (OSError, ValueError, KeyError) as error:
        st.warning("Verified offline monitoring reports are unavailable.")
        st.caption(str(error))
