import streamlit as st
import pandas as pd
from collections import defaultdict
import spacy
from spacy import displacy
import sys
import os
# 👇 importa tu función del notebook o módulo
sys.path.append(os.path.dirname(__file__))
from bio.prueba_modelo import ner_prediction, highlight_entities


st.set_page_config(page_title="MedNERDS", layout="wide")

st.title("🧬 MedNERDs App")
st.markdown("Extracción de entidades médicas")

# ==========================
# INPUT UI
# ==========================

text = st.text_area(
    "Introduce texto clínico:",
    height=180,
    placeholder="Ej: Patient diagnosed with diabetes and prescribed ibuprofen..."
)

run = st.button("Analizar")

# ==========================
# PROCESS
# ==========================

if run and text.strip():
    
    result = ner_prediction(text, compute="gpu")

    # convertir DataFrame → lista de dicts adaptados
    entities = []

    for _, row in result.iterrows():
        entities.append({
            "type": row["entity_group"],
            "value": row["value"],
            "start": row["start"],
            "end": row["end"],
            "score": float(row["score"]),
        })

    from collections import defaultdict

    grouped = defaultdict(list)

    for ent in entities:
        grouped[ent["type"]].append(ent["value"])
        counts = {k: len(v) for k, v in grouped.items()}


    # ==========================
    # Texto con Highlights
    # ==========================
    st.subheader("📄 Texto introducido")
    html = highlight_entities(text, entities)
    st.markdown(html, unsafe_allow_html=True) 

    col1, col2 = st.columns(2)

    # ==========================
    # DETALLE ENTIDADES
    # ==========================


    with col1:
        st.subheader("🔎 Entidades detectadas")

        if entities:
            df = pd.DataFrame(entities)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No se detectaron entidades con ese umbral.")

    # ==========================
    # AGRUPACIÓN
    # ==========================

    with col2:
        st.subheader("📊 Agrupación por tipo")

        if grouped:
            df_grouped = pd.DataFrame(
                [(k, v, counts[k]) for k, v in grouped.items()],
                columns=["Tipo", "Valores", "Cantidad"]
            )
            st.dataframe(df_grouped, use_container_width=True, hide_index=True)
        else:
            st.info("Sin datos agrupados.")

    # ==========================
    # JSON RAW (debug útil)
    # ==========================

    st.markdown("---")
    st.subheader("🧾 Output estructurado")
    st.text_area("Resultados", value=str(result.to_dict(orient="records")), height=200)

# ==========================
# FOOTER
# ==========================

st.markdown("---")
st.caption("Biomedical MedNERDs · PyCamp 2026")