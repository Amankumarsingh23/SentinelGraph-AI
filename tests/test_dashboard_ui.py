"""Execute the Streamlit views and verify transaction selection changes evidence."""

from pathlib import Path
import unittest
import pytest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.evidence import deterministic_summary, load_evidence_table, make_evidence_record

APP = Path(__file__).resolve().parents[1] / "dashboard/app.py"


@pytest.mark.private
class DashboardUiTests(unittest.TestCase):
    def test_invalid_evidence_stops_the_investigator_view(self):
        st.cache_data.clear()
        with patch("dashboard.evidence.load_evidence_table", side_effect=ValueError("Evidence mismatch")):
            app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertEqual(len(app.metric), 0)

    def test_invalid_shap_is_hidden_while_matched_transaction_remains_visible(self):
        st.cache_data.clear()
        with patch("dashboard.shap_evidence.load_shap_export", side_effect=ValueError("Different model hash")):
            app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 1)
        self.assertEqual(len(app.metric), 3)
        self.assertIn("No fraud decision has been assigned", app.info[0].value)

    def test_selection_changes_score_amount_features_and_summary_without_labels(self):
        table = load_evidence_table()
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        first = int(table.TransactionID.iloc[0])
        other = int(table.TransactionID.iloc[-1])
        self.assertEqual(app.selectbox(key="selected_transaction").value, first)
        app.selectbox(key="selected_transaction").set_value(other).run()
        self.assertFalse(app.exception)
        evidence = make_evidence_record(table, other)
        self.assertEqual(app.metric[0].value, f"{evidence['model_score']:.6f}")
        self.assertEqual(app.metric[1].value, f"{evidence['transaction_amount']:,.2f}")
        self.assertEqual(app.info[0].value, deterministic_summary(evidence))
        shown = app.dataframe[1].value
        self.assertEqual(dict(zip(shown['Recorded feature'], shown['Value'])), evidence['historical_features'])
        for item in app.dataframe:
            self.assertNotIn("isFraud", item.value.columns)
        self.assertFalse(app.warning)

    def test_retrospective_view_is_separate_and_outcomes_require_opt_in(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.radio(key="view").set_value("Retrospective clusters").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "9")
        self.assertEqual(app.metric[1].value, "87")
        self.assertNotIn("fraud_rate", app.dataframe[0].value)
        app.checkbox(key="retrospective_outcomes").check().run()
        self.assertIn("fraud_rate", app.dataframe[0].value)
        self.assertFalse(app.exception)

    def test_monitoring_view_displays_matching_saved_drift(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.radio(key="view").set_value("Monitoring").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.warning)
        self.assertEqual(app.metric[0].value, "0.305103")
        self.assertEqual(len(app.dataframe[0].value), 5)
