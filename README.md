# Zero-Shot Multilingual Code Comment Generation

[cite_start]This project focuses on generating human-readable descriptions (comments) for code snippets in multiple languages, using a zero-shot learning approach[cite: 1, 4, 7]. [cite_start]The goal is to provide tools for developers who work with non-English codebases, an area that is currently under-researched[cite: 8].

## Project Overview

[cite_start]The task is to take a function or code snippet and generate an accurate, fluent comment in a target language like Hindi, Tamil, Spanish, French, German, and others[cite: 7, 13]. [cite_start]We use a "zero-shot" method, meaning the models are leveraged without being explicitly fine-tuned on paired code-comment examples in these specific target languages[cite: 7, 35].

### Key Features

* [cite_start]**Multilingual Dataset:** We created a custom dataset of ~1500 examples by extending the XLCoST dataset[cite: 11, 16, 23].
* [cite_start]**Data Augmentation:** We used the Google Translate API and LLM synthesis to translate existing English comments and create new code-comment pairs for our target languages[cite: 14, 15, 31].
* [cite_start]**Models:** The project experiments with powerful pre-trained models like **CodeT5** and **PLBART**, which are designed for code intelligence tasks[cite: 27, 29].
* [cite_start]**Data Validation:** A comprehensive data validation pipeline (see `Project_Data_validation.ipynb`) was used to clean the data, check for missing values, analyze language distributions, and generate embeddings[cite: 37, 43, 56, 61].
* [cite_start]**Evaluation:** We evaluated the generated comments using automatic metrics like BLEU and ROUGE, as well as qualitative manual review, which confirmed that the models produce fluent and correct descriptions in most cases[cite: 96, 99, 102, 103].

## Dataset

[cite_start]The final dataset (`multilingual_code_comments.csv` / `.parquet`) was built from XLCoST and augmented translations[cite: 11, 14].

* [cite_start]**Target Languages:** Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn), Malayalam (ml), French (fr), German (de), Italian (it), Spanish (es), and English (en)[cite: 13].
* [cite_start]**Preprocessing:** Code and comments were normalized by converting to lowercase, stripping punctuation, and replacing code literals with placeholders (e.g., `<STR>`, `<NUM>`)[cite: 21, 22].
* [cite_start]**Sources:** Data is labeled by its origin (`XLCoST_Dataset`, `Generated_Data`) and intent (`Functionality`, `Translated`, `Synthetic`)[cite: 53, 81].

## How to Use

1.  **Dataset Creation:** The script `create_multilingual_code_comments_dataset.py` contains the logic for processing the base data and applying translations.
2.  **Data Validation:** The `Project_Data_validation.ipynb` notebook details the full pipeline for loading, cleaning, visualizing, and validating the dataset.
3.  [cite_start]**Models:** This project uses pre-trained CodeT5 and PLBART models for zero-shot inference[cite: 27, 29].

## Results

[cite_start]Both models achieved strong BLEU/ROUGE scores on English comments and promising scores on non-English languages[cite: 99, 100]. [cite_start]Manual inspection confirmed that generated comments were fluent and accurately preserved the meaning of the original code[cite: 102, 103].
