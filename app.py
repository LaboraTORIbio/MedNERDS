import streamlit as st
import pandas as pd
from collections import defaultdict

from bio.ner_prediction import ner_prediction, highlight_entities
from db import init_db, save_record, get_records_by_patient

init_db()

st.set_page_config(page_title="MedNERDS", layout="wide")

st.title("🧬 MedNERDS App")

tab1, tab2 = st.tabs(["📝 Insert", "🔍 Find"])

with tab1:
    # ==========================
    # INPUT UI
    # ==========================
    patient_id = st.text_input("Introduce patient ID:", placeholder="Ej: 12345")

    text = st.text_area(
        "Introduce clinical notes:",
        height=180,
        placeholder="Ej: Patient diagnosed with diabetes and prescribed ibuprofen...",
    )

    run = st.button("Analize")

    # ==========================
    # PROCESS
    # ==========================

    if run and text.strip() and patient_id.strip():
        pred_df, final_df = ner_prediction(text)

        # convertir DataFrame → lista de dicts adaptados
        entities = []

        for _, row in pred_df.iterrows():
            entities.append(
                {
                    "type": row["entity_group"],
                    "value": row["value"],
                    "start": row["start"],
                    "end": row["end"],
                    "score": float(row["score"]),
                }
            )

        grouped = defaultdict(list)

        for ent in entities:
            grouped[ent["type"]].append(ent["value"])
            counts = {k: len(v) for k, v in grouped.items()}

        # ==========================
        # Texto con Highlights
        # ==========================
        st.subheader("📄 Highlighted Entities")
        html = highlight_entities(text, entities)
        st.markdown(html, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # ==========================
        # DETALLE ENTIDADES
        # ==========================

        with col1:
            st.subheader("🔎 Recognized Entities")

            if entities:
                df = pd.DataFrame(entities)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No entities were detected at that threshold.")

        # ==========================
        # AGRUPACIÓN
        # ==========================

        with col2:
            st.subheader("📊 Grouped by type")

            if grouped:
                df_grouped = pd.DataFrame(
                    [(k, v, counts[k]) for k, v in grouped.items()],
                    columns=["Type", "Values", "Count"],
                )
                st.dataframe(df_grouped, use_container_width=True, hide_index=True)
            else:
                st.info("Ungrouped data.")

        # ==========================
        # JSON RAW (debug útil)
        # ==========================

        st.markdown("---")
        st.subheader("🧾 Structured Output")
        finalOutput = final_df.copy()
        finalOutput.insert(0, "Patient Id", patient_id)
        st.dataframe(finalOutput, use_container_width=True, hide_index=True)

        save_record(patient_id, final_df)

    with tab2:
        st.subheader("🔍 Search by patient ID")
        search_id = st.text_input(
            "Enter patient ID to search:", placeholder="Ej: 12345"
        )
        search_btn = st.button("Search")

        if search_btn and search_id.strip():
            records = get_records_by_patient(search_id)
            if records:
                df = pd.DataFrame(records)
                df = df.rename(
                    columns={
                        "patient_id": "Patient ID",
                        "age": "Age",
                        "sex": "Sex",
                        "history": "History",
                        "symptoms": "Symptoms",
                        "medication": "Medication",
                        "created_at": "Date",
                    }
                )

                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No records found with that ID.")

# ==========================
# FOOTER
# ==========================

st.markdown("---")
st.caption("Biomedical MedNERDs · PyCamp 2026")
