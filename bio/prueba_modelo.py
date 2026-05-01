#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict


MODEL_NAME = "d4data/biomedical-ner-all"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
)


def extract_ner(text: str, min_score: float = 0.5) -> dict:
    raw_entities = ner_pipeline(text)

    entities = []
    grouped = defaultdict(list)

    for ent in raw_entities:
        if ent["score"] < min_score:
            continue

        entity = {
            "type": ent["entity_group"],
            "value": ent["word"],
            "start": ent["start"],
            "end": ent["end"],
            "score": round(float(ent["score"]), 4),
        }
        entities.append(entity)
        grouped[ent["entity_group"]].append(ent["word"])

    return {
        "text": text,
        "entities": entities,
        "grouped": dict(grouped),
        "counts": {k: len(v) for k, v in grouped.items()},
    }



# In[ ]:


texto = (
    "The patient is a 65-year-old male with a history of hypertension and "
    "type 2 diabetes mellitus. He presented with chest pain and shortness "
    "of breath. He was started on aspirin 81 mg daily and metoprolol 25 mg "
    "twice daily. CT scan revealed bilateral pulmonary embolism."
)

resultado = extract_ner(texto)


print(json.dumps(resultado, indent=2, ensure_ascii=False))

