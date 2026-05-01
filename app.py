import streamlit as st
import pandas as pd
from collections import defaultdict
import sys
import os
# 👇 importa tu función del notebook o módulo
sys.path.append(os.path.dirname(__file__))
from bio.prueba_modelo import extract_ner 

st.set_page_config(page_title="MedNERDS", layout="wide")

st.title("🧬 Biomedical MedNERDs App")
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

    result = extract_ner(text)

    entities = result["entities"]
    grouped = result["grouped"]
    counts = result["counts"]

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
    st.json(result)

# ==========================
# FOOTER
# ==========================

st.markdown("---")
st.caption("Biomedical MedNERDs · PyCamp 2026")