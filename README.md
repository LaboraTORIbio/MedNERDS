# MedNERDS
MedNERds (**Med**ical **N**amed **E**ntity **R**ecognition for **D**ata **S**tructuring) is a clinical natural language processing (NLP) project designed to **extract structured medical information from unstructured clinical text**. The goal is to convert free-text clinical narratives into structured, analyzable data. The system uses a **pre-trained NER model** based on the **BioBERT transformer** to identify relevant entities (e.g., medical history, symptoms, medications), and augments these predictions with post-processing steps such as **negation detection**.

## Features
* The **Streamlit app** allows the user to **input a medical note in free-text format**, along with a **patient identifier**.
* By clicking the **“Analyze”** button, the text is passed to a **BioBERT-based NER model** ([d4data/biomedical-ner-all](https://huggingface.co/d4data/biomedical-ner-all)), which detects entities such as patient **age**, **sex**, **medical history**, **symptoms**, and **medications**. The [NegSpaCy](https://spacy.io/universe/project/negspacy) library is also used to **identify whether detected entities are negated**.
* The **entities** identified by the model **can be visualized** in the app as **highlighted spans** within the input text.
* The output of the NER model is then converted into a **structured table**, which is **also displayed** in the app.
* The **output table** is automatically appended to a **SQLite database**, along with the date and time of the record.
* In the **“Find”** tab, the user can **search for records** of a specific patient by entering the patient identifier.

<img src="./docs/resources/demo_for_readme.gif" width="1000" alt="Streamlit App">

## How to run the app
Install dependencies with *uv*:
```
uv sync
```

Run Streamlit app (**note**: models are downloaded on first launch, which may take a few minutes):
```
uv run streamlit run app.py
```

## Future steps
* **Fine-tune a custom NER model.** Adapt a transformer model such as ClinicalBERT to a specific use case by training on a curated corpus of EHR data. This would allow to define custom entity types (e.g., comorbidities, lab values, procedures) and improve performance on domain-specific language compared to general biomedical models.
* **Standardize extracted entities.** Map detected entities to controlled vocabularies such as SNOMED CT or UMLS. This enables consistent representation of clinical concepts, facilitates interoperability, and allows downstream analysis (e.g., grouping synonymous terms under the same concept ID).
* **Integrate PoS tagging and relation extraction.** Use part-of-speech tagging and relation extraction techniques to capture relationships between entities (e.g., medication–dosage, symptom–anatomical location). This moves the system beyond entity detection toward structured clinical understanding.
* **Add OCR for document ingestion.** Incorporate optical character recognition using tools like Tesseract OCR to process scanned PDFs or images. This would allow the pipeline to handle non-digital clinical documents and expand input sources.
* **Add multi-language support.**
* **Leverage large language models (LLMs).** Explore using LLMs for entity extraction, normalization, or validation. Models such as GPT-4 can complement traditional NER by handling ambiguous cases, improving recall, or assisting in tasks like relation extraction and summarization.

