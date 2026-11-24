"""MedStock Sentinel: synthetic inventory monitoring prototype."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, timedelta
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from database.supabase_client import InventoryStore
from modules.ai_recommendations import (
    CACHE_TTL_SECONDS,
    get_cached_recommendations,
    inventory_data_fingerprint,
    rule_based_recommendations,
)
from modules.config import setting
from modules.inventory import REQUIRED_COLUMNS, add_inventory_intelligence, load_inventory_csv, summarize_inventory, validate_inventory, InventoryValidationError
from modules.medicine_updates import changed_medicine_fields, discard_pending_update
from modules.anomaly_detection import (
    ALGORITHM as ANOMALY_ALGORITHM,
    FEATURE_COLUMNS as ANOMALY_FEATURES,
    MIN_TRAINING_RECORDS,
    MODEL_NAME as ANOMALY_MODEL_NAME,
    inventory_anomaly_results,
)

load_dotenv()
st.set_page_config(page_title="MedStock Sentinel", page_icon="💊", layout="wide")

RISK_COLORS = {"Critical": "#c62828", "High": "#ef6c00", "Moderate": "#f9a825", "Low": "#2e7d32", "Expired": "#7b1fa2", "Soon": "#ef6c00"}


@st.cache_resource
def store() -> InventoryStore:
    return InventoryStore()


@st.cache_resource
def gemini_executor() -> ThreadPoolExecutor:
    """Bounded worker pool; worker tasks never access Streamlit state or UI."""
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="medstock-gemini")


@st.fragment(run_every="1s")
def render_pending_gemini_result(current_inventory_fingerprint: str) -> None:
    future: Future | None = st.session_state.get("gemini_job_future")
    if future is None:
        return
    if not future.done():
        st.info("Gemini is preparing an optional explanation in the background. Python recommendations and other inventory pages remain available.")
        return

    snapshot = st.session_state.get("gemini_job_inventory")
    job_fingerprint = st.session_state.get("gemini_job_fingerprint")
    try:
        result, used_gemini, notice, cache_hit, cache_entry, request_seconds, cache_age = future.result()
    except Exception as exc:
        result = rule_based_recommendations(snapshot)
        used_gemini, notice, cache_hit, cache_entry = False, None, False, None
        request_seconds, cache_age = 0.0, None
        notice = f"Gemini enhancement failed ({type(exc).__name__}). Python-generated recommendations remain available."

    stale = job_fingerprint != current_inventory_fingerprint
    if not stale:
        if cache_entry:
            st.session_state.gemini_recommendation_cache = cache_entry
        if used_gemini:
            st.session_state.pop("gemini_last_notice", None)
            st.session_state.gemini_last_result = {
                "text": result,
                "inventory_fingerprint": job_fingerprint,
                "created_at": time.monotonic(),
                "request_duration_seconds": float(
                    cache_entry.get("request_duration_seconds", request_seconds)
                    if cache_hit and cache_entry else request_seconds
                ),
            }
            st.markdown("#### Gemini explanation")
            st.markdown(result)
            if cache_hit:
                st.caption(f"Cached explanation reused; age {cache_age:.1f} seconds. No Gemini request was sent.")
            else:
                st.caption(f"Gemini enhancement completed in {request_seconds:.2f} seconds.")
        elif notice:
            st.session_state.gemini_last_notice = notice + " Python-generated recommendations remain available above."
        if not cache_hit:
            try:
                store().record_analysis(snapshot, result, "gemini" if used_gemini else "rules")
                st.caption("Recommendation history saved.")
            except RuntimeError as exc:
                st.warning(f"Recommendation history could not be saved ({type(exc).__name__}).")
    else:
        st.session_state.gemini_last_notice = "Inventory changed while Gemini was processing. The stale explanation was discarded; current Python recommendations remain available."
        if not cache_hit:
            try:
                store().record_analysis(snapshot, result, "gemini" if used_gemini else "rules")
            except RuntimeError:
                pass

    st.session_state.pop("gemini_job_future", None)
    st.session_state.pop("gemini_job_inventory", None)
    st.session_state.pop("gemini_job_fingerprint", None)
    st.rerun()


def init_data() -> None:
    if "inventory" not in st.session_state:
        try:
            st.session_state.inventory = store().load()
        except Exception as exc:
            st.session_state.inventory = pd.DataFrame(columns=REQUIRED_COLUMNS)
            st.session_state.load_error = str(exc)


def persist(frame: pd.DataFrame) -> bool:
    try:
        clean = validate_inventory(frame)
        store().save(clean)
        st.session_state.inventory = clean
        st.success("Inventory saved.")
        return True
    except (InventoryValidationError, RuntimeError, OSError) as exc:
        st.error(str(exc))
    return False


def render_update_medicine(store_obj: InventoryStore) -> None:
    """Render a staged ID lookup and single-record update workflow."""
    st.subheader("Update Medicine")
    st.caption("Search by Medicine ID first. Editing is available only after a matching Supabase record is found.")
    flash = st.session_state.pop("update_notice", None)
    if flash:
        st.success(flash)
    with st.form("medicine_id_lookup"):
        lookup_id = st.text_input("Medicine ID", key="update_lookup_id", placeholder="For example, MED001")
        search_clicked = st.form_submit_button("Search")
    if search_clicked:
        discard_pending_update(st.session_state)
        st.session_state.update_lookup_error = None
        medicine_id = lookup_id.strip()
        if not medicine_id:
            st.session_state.update_lookup_error = "Enter a Medicine ID to search."
        else:
            try:
                found = store_obj.get_medicine(medicine_id)
                if found is None:
                    st.session_state.update_lookup_error = f"No medicine was found with ID {medicine_id}. Check the ID and try again."
                else:
                    st.session_state.update_record = found
                    st.session_state.update_lookup_error = None
            except (RuntimeError, ValueError) as exc:
                st.session_state.update_lookup_error = f"Could not search Supabase. Check the cloud connection and try again. ({type(exc).__name__})"
    if st.session_state.get("update_lookup_error"):
        st.error(st.session_state.update_lookup_error)

    record = st.session_state.get("update_record")
    if record is None:
        return
    generation = int(st.session_state.get("update_generation", 0))
    st.markdown(f"**Selected medicine:** {record['medicine_name']}  \n**Medicine ID:** `{record['medicine_id']}`")
    with st.form(f"medicine_update_form_{generation}"):
        st.text_input("Medicine ID (read-only)", value=record["medicine_id"], disabled=True, key=f"update_id_{generation}")
        a, b, c = st.columns(3)
        proposed = {
            "medicine_name": a.text_input("Medicine name", value=record["medicine_name"], key=f"update_name_{generation}"),
            "category": b.text_input("Category", value=record["category"], key=f"update_category_{generation}"),
            "supplier_name": c.text_input("Supplier name", value=record["supplier_name"], key=f"update_supplier_{generation}"),
            "current_stock": a.number_input("Current stock", min_value=0.0, value=float(record["current_stock"]), step=1.0, key=f"update_stock_{generation}"),
            "average_daily_consumption": b.number_input("Average daily consumption", min_value=0.0, value=float(record["average_daily_consumption"]), step=1.0, key=f"update_consumption_{generation}"),
            "supplier_lead_time": c.number_input("Supplier lead time (days)", min_value=0.0, value=float(record["supplier_lead_time"]), step=1.0, key=f"update_lead_{generation}"),
            "safety_stock": a.number_input("Safety stock", min_value=0.0, value=float(record["safety_stock"]), step=1.0, key=f"update_safety_{generation}"),
            "reorder_quantity": b.number_input("Reorder quantity", min_value=0.0, value=float(record["reorder_quantity"]), step=1.0, key=f"update_reorder_{generation}"),
            "manufacturing_date": c.date_input("Manufacturing date", value=date.fromisoformat(str(record["manufacturing_date"])[:10]), key=f"update_manufacturing_{generation}"),
            "expiry_date": a.date_input("Expiry date", value=date.fromisoformat(str(record["expiry_date"])[:10]), key=f"update_expiry_{generation}"),
        }
        save_clicked = st.form_submit_button("Save Changes", type="primary")
        cancel_clicked = st.form_submit_button("Cancel")
    if cancel_clicked:
        discard_pending_update(st.session_state)
        st.session_state.update_lookup_error = None
        st.rerun()
    if save_clicked:
        try:
            changes = changed_medicine_fields(record, proposed)
            if not changes:
                st.info("No changes to save.")
                return
            updated = store_obj.update_medicine(record["medicine_id"], changes)
            if updated is None:
                discard_pending_update(st.session_state)
                st.error("This medicine no longer exists in Supabase. Search for it again before editing.")
                return
            try:
                st.session_state.inventory = store_obj.load()
                notice = f"{updated['medicine_name']} ({updated['medicine_id']}) was updated successfully."
            except RuntimeError:
                current_inventory = st.session_state.inventory.copy()
                row_mask = current_inventory["medicine_id"] == record["medicine_id"]
                for field in REQUIRED_COLUMNS:
                    if field in updated:
                        current_inventory.loc[row_mask, field] = updated[field]
                st.session_state.inventory = validate_inventory(current_inventory)
                notice = f"{updated['medicine_name']} ({updated['medicine_id']}) was saved, but the dashboard refresh failed. Reload the page to refresh all inventory."
            st.session_state.update_record = updated
            st.session_state.update_notice = notice
            st.rerun()
        except InventoryValidationError as exc:
            st.error(f"Please correct the medicine details: {exc}")
        except (RuntimeError, ValueError) as exc:
            st.error(f"Could not save this medicine to Supabase. No other medicine records were changed. ({type(exc).__name__})")


init_data()
inventory = st.session_state.inventory
store_obj = store()

with st.sidebar:
    st.title("💊 MedStock Sentinel")
    st.caption("Medicine inventory early warning")
    page = st.radio("Navigate", ["Dashboard", "Inventory management", "Recommendations"], label_visibility="collapsed")
    st.divider()
    if store_obj.cloud:
        if store_obj.reachable:
            st.success("CLOUD · Supabase connected")
        else:
            st.warning("CLOUD CONFIGURED · connection not verified")
    else:
        st.info("LOCAL DEMONSTRATION MODE · CSV storage")
    if setting("GEMINI_API_KEY"):
        st.caption("Gemini API key configured · live response checked on request")
    else:
        st.caption("Gemini unavailable · rule-based recommendations active")
    if st.button("Check Supabase connection", disabled=not store_obj.cloud):
        try:
            store_obj.check_connection()
            st.success("Supabase connection verified.")
        except RuntimeError as exc:
            st.error(str(exc))
    st.caption("Academic prototype · synthetic inventory only · not medical advice")
    if store_obj.error:
        st.warning(store_obj.error)
    if store_obj.config_warning:
        st.info(store_obj.config_warning)
    if st.session_state.get("load_error"):
        st.error(f"Could not load inventory: {st.session_state.load_error}")

st.title(page)
st.caption("Operational inventory support only. Risk flags are explainable stock heuristics, not medical advice.")

if page == "Dashboard":
    calculated = add_inventory_intelligence(inventory) if len(inventory) else add_inventory_intelligence(pd.DataFrame(columns=REQUIRED_COLUMNS))
    metrics = summarize_inventory(inventory)
    cols = st.columns(5)
    labels = [("Total medicines", metrics["total_medicines"]), ("Inventory units", f"{metrics['total_quantity']:,}"), ("Low stock", metrics["low_stock_medicines"]), ("Critical alerts", metrics["critical_alerts"]), ("Expiry risks", metrics["expiring_medicines"])]
    for col, (label, value) in zip(cols, labels):
        col.metric(label, value)
    if inventory.empty:
        st.info("No inventory records found. Add medicines or upload a CSV in Inventory management.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Stock by medicine")
            chart = calculated.sort_values("current_stock", ascending=False)
            st.plotly_chart(px.bar(chart, x="medicine_name", y="current_stock", color="stockout_risk", color_discrete_map=RISK_COLORS, labels={"medicine_name": "Medicine", "current_stock": "Units", "stockout_risk": "Stock risk"}), width="stretch")
        with right:
            st.subheader("Risk distribution")
            counts = calculated["stockout_risk"].value_counts().rename_axis("Risk").reset_index(name="Medicines")
            st.plotly_chart(px.pie(counts, names="Risk", values="Medicines", color="Risk", color_discrete_map=RISK_COLORS, hole=.45), width="stretch")
        st.subheader("Inventory early warnings")
        shown = calculated[calculated["stockout_risk"].isin(["Critical", "High", "Moderate"]) | calculated["expiry_risk"].isin(["Expired", "Critical", "Soon"])].copy()
        st.dataframe(shown[["medicine_id", "medicine_name", "current_stock", "days_until_stockout", "reorder_point", "stockout_risk", "expiry_date", "expiry_risk", "suggested_reorder_quantity"]], width="stretch", hide_index=True)

elif page == "Inventory management":
    st.caption("All sample rows are synthetic. Upload CSV with the exact column headers shown in the template.")
    with st.expander("Add a medicine", expanded=False):
        with st.form("add_medicine", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            mid = c1.text_input("Medicine ID")
            name = c2.text_input("Medicine name")
            category = c3.text_input("Category")
            stock = c1.number_input("Current stock (units)", min_value=0, step=1)
            consumption = c2.number_input("Average daily consumption", min_value=0.0, step=1.0)
            lead = c3.number_input("Supplier lead time (days)", min_value=0, step=1)
            safety = c1.number_input("Safety stock", min_value=0, step=1)
            reorder = c2.number_input("Reorder quantity", min_value=0, step=1)
            supplier = c3.text_input("Supplier name")
            manufacturing = c1.date_input("Manufacturing date", value=date.today())
            expiry = c2.date_input("Expiry date", value=date.today() + timedelta(days=365))
            submitted = st.form_submit_button("Add medicine")
            if submitted:
                new = {"medicine_id": mid, "medicine_name": name, "category": category, "current_stock": stock, "average_daily_consumption": consumption, "supplier_lead_time": lead, "safety_stock": safety, "reorder_quantity": reorder, "manufacturing_date": manufacturing, "expiry_date": expiry, "supplier_name": supplier}
                try:
                    candidate = validate_inventory(pd.concat([inventory, pd.DataFrame([new])], ignore_index=True))
                    if persist(candidate):
                        st.rerun()
                except InventoryValidationError as exc:
                    st.error(str(exc))
    render_update_medicine(store_obj)
    with st.expander("Upload inventory CSV"):
        st.download_button("Download CSV template", pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(index=False), "inventory_template.csv", "text/csv")
        upload = st.file_uploader("Select inventory CSV", type=["csv"])
        mode = st.radio("Import behavior", ["Replace inventory", "Append to inventory"], horizontal=True)
        if upload and st.button("Import CSV"):
            try:
                incoming = load_inventory_csv(upload)
                combined = incoming if mode == "Replace inventory" else pd.concat([inventory, incoming], ignore_index=True)
                if persist(combined):
                    st.rerun()
            except (InventoryValidationError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"CSV import failed: {exc}")
    st.subheader("Search, edit, and delete inventory")
    st.caption("Search by medicine, supplier, or ID; narrow by category and calculated stock risk. For a guided edit, use Update Medicine above.")
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search inventory")
    categories = ["All"] + sorted(inventory["category"].unique().tolist()) if not inventory.empty else ["All"]
    category_filter = f2.selectbox("Category", categories)
    risk_options = ["All", "Critical", "High", "Moderate", "Low"]
    risk_filter = f3.selectbox("Stock risk", risk_options)
    visible = inventory.copy()
    if query:
        search_cols = ["medicine_id", "medicine_name", "category", "supplier_name"]
        mask = visible[search_cols].astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False)).any(axis=1)
        visible = visible[mask]
    if category_filter != "All":
        visible = visible[visible["category"] == category_filter]
    if risk_filter != "All" and not visible.empty:
        risk_by_id = add_inventory_intelligence(inventory).set_index("medicine_id")["stockout_risk"]
        visible = visible[visible["medicine_id"].map(risk_by_id) == risk_filter]
    edited = st.data_editor(visible, num_rows="fixed", width="stretch", hide_index=True, key="inventory_editor", disabled=["medicine_id"])
    edit_col, delete_col = st.columns(2)
    if edit_col.button("Save table edits"):
        changed_rows = []
        try:
            originals = inventory.set_index("medicine_id")
            for _, row in edited.iterrows():
                medicine_id = str(row["medicine_id"])
                if medicine_id not in originals.index:
                    raise InventoryValidationError("A medicine ID changed while editing. Reload the inventory and try again.")
                current_record = originals.loc[medicine_id].to_dict()
                proposed_fields = row.drop(labels=["medicine_id"]).to_dict()
                changes = changed_medicine_fields(current_record, proposed_fields)
                if changes:
                    changed_rows.append((medicine_id, changes))
            if not changed_rows:
                st.info("No changes to save.")
            else:
                saved_count = 0
                for medicine_id, changes in changed_rows:
                    if store_obj.update_medicine(medicine_id, changes) is None:
                        raise RuntimeError("A medicine was removed before its update could be saved.")
                    saved_count += 1
                st.session_state.inventory = store_obj.load()
                st.session_state.update_notice = f"Saved edits to {saved_count} selected medicine record(s)."
                st.rerun()
        except InventoryValidationError as exc:
            st.error(f"Please correct the medicine details: {exc}")
        except (RuntimeError, ValueError) as exc:
            st.error(f"Some table edits could not be saved to Supabase. Changes saved before the error remain applied. ({type(exc).__name__})")
    delete_ids = st.multiselect("Medicine IDs to delete", inventory["medicine_id"].tolist() if not inventory.empty else [])
    if delete_col.button("Delete selected", disabled=not delete_ids):
        persist(inventory[~inventory["medicine_id"].isin(delete_ids)].reset_index(drop=True))
        st.rerun()

    with st.expander("Record medicine consumption", expanded=False):
        if inventory.empty:
            st.info("Add a medicine before recording consumption.")
        else:
            with st.form("record_consumption", clear_on_submit=True):
                medicine_id = st.selectbox("Medicine", inventory["medicine_id"].tolist(), format_func=lambda value: f"{value} · {inventory.loc[inventory['medicine_id'] == value, 'medicine_name'].iloc[0]}")
                available = float(inventory.loc[inventory["medicine_id"] == medicine_id, "current_stock"].iloc[0])
                st.caption(f"Available stock: {available:g}")
                quantity = st.number_input("Quantity consumed", min_value=0.01, max_value=max(0.01, available), value=min(1.0, max(0.01, available)), step=1.0, disabled=available <= 0)
                consumed_on = st.date_input("Consumption date", value=date.today())
                note = st.text_input("Optional operational note")
                record = st.form_submit_button("Save consumption", disabled=available <= 0)
                if record:
                    if quantity > available:
                        st.error("Consumption cannot exceed available stock.")
                    else:
                        try:
                            medicine_name = str(inventory.loc[inventory["medicine_id"] == medicine_id, "medicine_name"].iloc[0])
                            store_obj.record_consumption(medicine_id, quantity, consumed_on.isoformat(), note, medicine_name)
                            st.session_state.inventory = store_obj.load()
                            st.success("Consumption history saved and stock updated.")
                            st.rerun()
                        except (RuntimeError, ValueError, OSError) as exc:
                            st.error(str(exc))
    with st.expander("Recent consumption history"):
        try:
            history = store_obj.load_consumption_history()
            if history:
                st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
            else:
                st.caption("No consumption entries yet.")
        except RuntimeError as exc:
            st.warning(str(exc))

else:
    st.subheader("AI inventory recommendations")
    st.write("Python calculates and displays inventory recommendations immediately. Gemini adds an optional qualitative explanation. Exact quantities, dates, risk labels, and calculations remain Python-generated evidence.")
    evidence = add_inventory_intelligence(inventory)
    current_fingerprint = inventory_data_fingerprint(evidence)
    st.markdown("#### Python-generated recommendations")
    st.markdown(rule_based_recommendations(evidence))
    st.caption("Available immediately from the current inventory, even when Gemini is unavailable.")
    if not inventory.empty:
        with st.expander("Python-calculated inventory evidence", expanded=False):
            st.dataframe(evidence[["medicine_id", "medicine_name", "current_stock", "days_until_stockout", "reorder_point", "stockout_risk", "expiry_date", "expiry_risk", "suggested_reorder_quantity"]], width="stretch", hide_index=True)

    st.markdown("#### Prototype ML inventory anomaly detection")
    st.caption(
        f"Model: {ANOMALY_MODEL_NAME} · Algorithm: {ANOMALY_ALGORITHM} · "
        f"Features: {', '.join(ANOMALY_FEATURES)}"
    )
    try:
        anomaly_results, training_records = inventory_anomaly_results(evidence)
        st.caption(f"Training records used: {training_records}. Prototype unsupervised model; scores are relative anomaly scores, not probabilities or accuracy metrics.")
        if training_records < MIN_TRAINING_RECORDS:
            st.info(f"At least {MIN_TRAINING_RECORDS} valid inventory records are needed to fit this prototype model. No anomaly predictions are shown.")
        elif anomaly_results.empty:
            st.info("No valid inventory records are available for model training.")
        else:
            detected = anomaly_results[anomaly_results["status"] == "Anomalous"]
            if detected.empty:
                st.success("The model did not flag any anomalous records in this inventory.")
            else:
                st.warning(f"The model flagged {len(detected)} anomalous record(s). Review these inventory values; this is a screening signal, not medical advice.")
                st.plotly_chart(
                    px.bar(
                        detected.sort_values("anomaly_score", ascending=True),
                        x="anomaly_score", y="medicine_name", orientation="h",
                        color="anomaly_score", color_continuous_scale="OrRd",
                        hover_data=["medicine_id", "explanation"],
                        labels={"anomaly_score": "Anomaly score (higher = more unusual)", "medicine_name": "Medicine"},
                    ),
                    width="stretch",
                )
            st.dataframe(anomaly_results, width="stretch", hide_index=True)
    except (ValueError, TypeError) as exc:
        st.warning(f"Prototype anomaly model could not process the current inventory: {exc}")

    api_key = setting("GEMINI_API_KEY")
    pending_future = st.session_state.get("gemini_job_future")
    if not api_key:
        st.info("Gemini is not configured. Python-generated recommendations are available above.")
    elif st.button(
        "Request Gemini explanation",
        type="primary",
        disabled=pending_future is not None or inventory.empty,
        key="request_gemini_explanation",
    ):
        st.session_state.pop("gemini_last_notice", None)
        snapshot = evidence.copy(deep=True)
        st.session_state.gemini_job_inventory = snapshot
        st.session_state.gemini_job_fingerprint = current_fingerprint
        st.session_state.gemini_job_future = gemini_executor().submit(
            get_cached_recommendations,
            snapshot,
            dict(st.session_state["gemini_recommendation_cache"])
            if st.session_state.get("gemini_recommendation_cache") else None,
            api_key=api_key,
            model=setting("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
        )

    if st.session_state.get("gemini_last_notice"):
        st.warning(st.session_state.gemini_last_notice)

    last_ai_result = st.session_state.get("gemini_last_result")
    if (
        last_ai_result
        and last_ai_result.get("inventory_fingerprint") == current_fingerprint
        and time.monotonic() - float(last_ai_result.get("created_at", 0)) < CACHE_TTL_SECONDS
    ):
        st.markdown("#### Gemini explanation")
        st.markdown(last_ai_result["text"])
        st.caption(f"Cached Gemini response · originally generated in {last_ai_result['request_duration_seconds']:.2f} seconds.")
    render_pending_gemini_result(current_fingerprint)

    if inventory.empty:
        st.info("Add inventory to generate Python recommendations and request a Gemini explanation.")
    with st.expander("Recent recommendation history"):
        try:
            history = store_obj.load_analysis_history()
            if history:
                st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
            else:
                st.caption("No saved recommendations yet.")
        except RuntimeError as exc:
            st.warning(str(exc))

st.divider()
st.caption("Stockout heuristic: stock ÷ average daily use; reorder point: lead-time demand + safety stock. Suggested order tops up to reorder point + configured order quantity.")
