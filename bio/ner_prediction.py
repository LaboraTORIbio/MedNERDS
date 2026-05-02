import re
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import spacy
import negspacy
from spacy.tokens import Span
from negspacy.negation import Negex


MODEL_NAME = "d4data/biomedical-ner-all"


def clean_corpus(corpus):
    cleaned_corpus = corpus.replace(".", " ")
    cleaned_corpus = cleaned_corpus.replace(",", " ")
    return cleaned_corpus


def detect_negations(corpus, df):
    if df.empty:
        return []

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("negex")

    doc = nlp(corpus)

    # Build spans
    spans = [
        doc.char_span(
            int(row["start"]),
            int(row["end"]),
            label=row["entity_group"],
            alignment_mode="expand",
        )
        for _, row in df.iterrows()
    ]

    doc.ents = [s for s in spans if s is not None]

    # Run negation AFTER setting entities
    doc = nlp.get_pipe("negex")(doc)

    # Extract negation flags, preserving order
    negated_flags = []
    ent_iter = iter(doc.ents)

    for span in spans:
        if span is None:
            negated_flags.append(False)
        else:
            negated_flags.append(next(ent_iter)._.negex)

    return negated_flags


def extract_age(value):
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def aggregate_entities(df):
    result = {"Age": None, "Sex": None, "History": [], "Symptoms": [], "Medication": []}

    if df.empty:
        return pd.DataFrame([result])

    for group, subdf in df.groupby("entity_group"):
        values = []
        for _, row in subdf.iterrows():
            v = row["value"]
            if row.get("negated", False):
                v = f"{v} (negated)"
            values.append(v)

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

    pred_df["negated"] = detect_negations(corpus, pred_df)

    actual_values_list = []
    for _, pred_df_row in pred_df.iterrows():
        actual_word = corpus[pred_df_row["start"] : pred_df_row["end"]]
        actual_values_list.append(actual_word)

    pred_df["value"] = actual_values_list

    if len(pred_df) != 0:
        pred_df = pred_df[
            ["entity_group", "value", "word", "start", "end", "score", "negated"]
        ]
        pred_df["entity_group"] = pred_df["entity_group"].replace(
            {"Sign_symptom": "Symptoms"}
        )
        pred_df = pred_df.drop_duplicates(
            subset=["entity_group", "value", "negated"], keep="first"
        ).reset_index(drop=True)

    final_df = aggregate_entities(pred_df)
    final_df = final_df.reindex(
        columns=["Age", "Sex", "History", "Symptoms", "Medication"]
    )

    return pred_df, final_df


def highlight_entities(text, entities):
    entities = sorted(entities, key=lambda x: x["start"])

    colors = {
        "Disease": "#e6af49",
        "Medication": "#6eade0",
        "Symptoms": "#f57969",
        "History": "#63e6be",
        "Default": "#c769ab",
    }

    html_parts = []
    last_idx = 0

    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["type"]

        color = colors.get(label, colors["Default"])

        html_parts.append(text[last_idx:start])

        html_parts.append(
            f'<span style="background-color:{color};padding:2px 4px;border-radius:4px;font-weight:bold;">'
            f"{text[start:end]} ({label})</span>"
        )

        last_idx = end

    html_parts.append(text[last_idx:])
    return "".join(html_parts)
