#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

import nltk
from nltk.tokenize import word_tokenize


# In[ ]:


MODEL_NAME = "d4data/biomedical-ner-all"

nltk.download('punkt')
nltk.download('punkt_tab')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)


# In[ ]:


def ner_prediction(corpus, compute):
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="max",
    )

    for sentence in sent_tokenizer.tokenize(corpus):
        pred = ner_pipeline(corpus)

        pred_df = pd.DataFrame(pred)

        actual_values_list = []
        for _, pred_df_row in pred_df.iterrows():
            actual_word = corpus[pred_df_row['start']: pred_df_row['end']]
            actual_values_list.append(actual_word)

        pred_df['value'] = actual_values_list

        if len(pred_df) != 0:
            final_df = pred_df[['entity_group', 'value', 'word', 'start', 'end', 'score']]

            final_df = final_df.drop_duplicates(
                subset = ['entity_group', 'value'],
                keep = 'first'
        ).reset_index(drop=True)

    return final_df


def highlight_entities(text, entities):
    entities = sorted(entities, key=lambda x: x["start"])

    colors = {
        "Disease": "#e6af49",
        "Medication": "#6eade0",
        "Sign_symptom": "#f57969",
        "History": "#63e6be",
        "Diagnostic_procedure": "#b197fc",
        "Biological_structure": "#a970cf",
        "Default": "#c769ab"
    }

    html = ""
    last_idx = 0

    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["type"]

        color = colors.get(label, colors["Default"])

        # texto normal antes de entidad
        html += text[last_idx:start]

        # entidad coloreada
        html += f"""
        <span style="
            background-color:{color};
            padding:2px 4px;
            border-radius:4px;
            color:black;
            font-weight:bold;
        ">
            {text[start:end]} ({label})
        </span>
        """

        last_idx = end

    # resto del texto
    html += text[last_idx:]

    return html


# In[ ]:


texto = (
    "The patient is a 65-year-old male with a history of hypertension and "
    "type 2 diabetes mellitus. He presented with chest pain and shortness "
    "of breath. He was started on aspirin 81 mg daily and metoprolol 25 mg "
    "twice daily. CT scan revealed bilateral pulmonary embolism."
)

ner_prediction(texto, compute="gpu")

