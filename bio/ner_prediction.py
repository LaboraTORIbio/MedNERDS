import re
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification


MODEL_NAME = "d4data/biomedical-ner-all"


def clean_corpus(corpus):
    cleaned_corpus = corpus.replace(".", " ")
    return cleaned_corpus


def extract_age(value):
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def aggregate_entities(df):
    result = {
        "Age": None,
        "Sex": None,
        "History": [],
        "Symptoms": [],
        "Medication": []
    }

    if df.empty:
        return pd.DataFrame([result])

    for group, subdf in df.groupby("entity_group"):
        values = subdf["value"].tolist()

        if group == "Age":
            ages = [extract_age(v) for v in values if extract_age(v) is not None]
            result[group] = ages[0] if ages else None

        elif group == "Sex":
            result[group] = values[0] if values else None

        else:
            seen = set()
            unique_values = []
            for v in values:
                if v not in seen:
                    seen.add(v)
                    unique_values.append(v)

            result[group] = unique_values

    return pd.DataFrame([result])


def ner_prediction(corpus):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="max",
    )

    cleaned_corpus = clean_corpus(corpus)

    pred = ner_pipeline(cleaned_corpus)

    pred_df = pd.DataFrame(pred)
    
    actual_values_list = []
    for _, pred_df_row in pred_df.iterrows():
        actual_word = corpus[pred_df_row['start']: pred_df_row['end']]
        actual_values_list.append(actual_word)

    pred_df['value'] = actual_values_list
    
    if len(pred_df) != 0:
        pred_df = pred_df[['entity_group', 'value', 'word', 'start', 'end', 'score']]
        pred_df['entity_group'] = pred_df['entity_group'].replace({"Sign_symptom": "Symptoms"})
        pred_df = pred_df.drop_duplicates(
            subset=['entity_group', 'value'],
            keep='first'
        ).reset_index(drop=True)
        
    final_df = aggregate_entities(pred_df)
    final_df = final_df.reindex(columns=["Age", "Sex", "History", "Symptoms", "Medication"])

    return pred_df, final_df


def highlight_entities(text, entities):
    entities = sorted(entities, key=lambda x: x["start"])

    colors = {
        "Disease": "#e6af49",
        "Medication": "#6eade0",
        "Symptoms": "#f57969",
        "History": "#63e6be",
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

