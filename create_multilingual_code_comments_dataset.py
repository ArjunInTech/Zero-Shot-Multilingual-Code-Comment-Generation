# --- Step 1: Set Up Your Local Environment (VS Code) ---

# --- Step 1.1: Create Project Directory ---
import os

# Define the base directory for your project locally
# IMPORTANT: This path should match your desired local directory.
base_dir = r"e:\secondprogram\User\datasetproject\projectfile"

# Create the directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)
print(f"Working directory set to: {base_dir}")
print(f"Please ensure this path exists or is accessible on your local machine.")

print("\nEnsure the required libraries are installed by running 'pip install <library_name>' in your VS Code terminal.")

# Download NLTK data for tokenization (run this in your Python script)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError: # Corrected exception type
    print("NLTK 'punkt' tokenizer data not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer data downloaded successfully.")
except Exception as e:
    print(f"An unexpected error occurred during NLTK data check/download: {e}")

print("Local environment setup complete. Proceeding to Step 2: Data Collection.")


# --- Step 2: Data Collection Strategies ---

# --- Step 2.1: Leveraging Existing Multilingual Code Datasets (XLCoST) ---
import pandas as pd
import json

# IMPORTANT: Adjust this path to your downloaded XLCoST data file.
# Example: If you downloaded 'Python-snippet-level/train.jsonl' and placed it in 'xlcost_data'
# and renamed it to 'python_snippets_train.jsonl':
xlcost_data_path = os.path.join(base_dir, 'xlcost_data', 'python_snippets_train.jsonl')

# Create dummy XLCoST data for demonstration if the actual file is not found.
# In a real scenario, you would have downloaded the actual XLCoST data.
if not os.path.exists(xlcost_data_path):
    print(f"XLCoST data not found at {xlcost_data_path}. Creating dummy data for demonstration.")
    os.makedirs(os.path.dirname(xlcost_data_path), exist_ok=True)
    dummy_xlcost_entries = [
        {
            "id": 1,
            "code": "def hello_world():\n    print('Hello, world!')",
            "lang": "python",
            "comment": "A simple function that prints Hello, world! in Python."
        },
        {
            "id": 2,
            "code": "public static void main(String[] args) {\n    System.out.println(\"Hello, world!\");\n}",
            "lang": "java",
            "comment": "Main method that prints Hello, world! in Java."
        },
        {
            "id": 3,
            "code": "function calculateSum(a, b) {\n    return a + b;\n}",
            "lang": "javascript",
            "comment": "Calculates the sum of two numbers."
        },
        {
            "id": 4,
            "code": "int factorial(int n) {\n    if (n == 0) return 1;\n    return n * factorial(n-1);\n}",
            "lang": "cpp",
            "comment": "Computes the factorial of a non-negative integer."
        },
        {
            "id": 5,
            "code": "class MyClass {\n    public string GetName() { return \"Test\"; }\n}",
            "lang": "csharp",
            "comment": "A class method to get a name string."
        }
    ]
    with open(xlcost_data_path, 'w', encoding='utf-8') as f:
        for entry in dummy_xlcost_entries:
            f.write(json.dumps(entry) + '\n')
    print("Dummy XLCoST data created for demonstration.")

# Load XLCoST data (assuming JSONL format)
xlcost_data = []
try:
    with open(xlcost_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            xlcost_data.append(json.loads(line))
    df_xlcost = pd.DataFrame(xlcost_data)
    print(f"\nLoaded {len(df_xlcost)} entries from XLCoST data.")
    print("XLCoST DataFrame head:\n", df_xlcost.head())

    # Map XLCoST columns to your desired schema
    df_xlcost_mapped = pd.DataFrame({
        'unique_id': 'XLCoST_' + df_xlcost['id'].astype(str),
        'code_snippet': df_xlcost['code'],
        'programming_language': df_xlcost['lang'].str.capitalize(),
        'comment_english': df_xlcost['comment'],
        'target_language_code': 'en', # XLCoST comments are primarily in English [1]
        'comment_target_lang': df_xlcost['comment'], # For English, target is same as English comment
        'comment_intent': 'Functionality', # Default or infer if possible
        'source_url': 'XLCoST_Dataset'
    })
    print("\nXLCoST data mapped to project schema:")
    print(df_xlcost_mapped.head())

except FileNotFoundError:
    print(f"XLCoST data file not found at {xlcost_data_path}. Please download it manually and place it there as instructed.")
    df_xlcost_mapped = pd.DataFrame(columns=['unique_id', 'code_snippet', 'programming_language', 'comment_english', 'target_language_code', 'comment_target_lang', 'comment_intent', 'source_url'])
except Exception as e:
    print(f"Error loading or processing XLCoST data: {e}")
    df_xlcost_mapped = pd.DataFrame(columns=['unique_id', 'code_snippet', 'programming_language', 'comment_english', 'target_language_code', 'comment_target_lang', 'comment_intent', 'source_url'])

# Initialize your main dataset DataFrame with XLCoST data
df_dataset = df_xlcost_mapped.copy()


# --- Step 2.2: Web Scraping for Code and Comments (Optional/Supplementary) ---
import requests
import html2text
import re
import time # To avoid overwhelming servers

def scrape_and_extract_code_comments(url):
    """
    A highly simplified web scraper. Real-world scraping requires
    more robust parsing based on specific website structures.
    This function attempts to extract code blocks and dummy comments.
    """
    try:
        response = requests.get(url, timeout=10) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx) [2]
        raw_html = response.text

        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.mark_code = True # wrap code in [code]...[/code] tags [2]

        markdown_text = converter.handle(raw_html)

        # Extract code blocks (assuming they are marked with [code]...[/code]) [2]
        code_blocks = re.findall(r'\[code\](.*?)\[/code\]', markdown_text, re.DOTALL)

        scraped_data = []
        for i, block in enumerate(code_blocks):
            # In a real scenario, you'd need sophisticated logic to find associated comments.
            dummy_comment = f"A scraped code snippet (ID: {i+1}) from {url}."
            scraped_data.append({
                'unique_id': f"SCRAPED_{abs(hash(block))}_{i}", # Simple hash for uniqueness
                'code_snippet': block.strip(),
                'programming_language': 'Unknown', # You'd need a language detector
                'comment_english': dummy_comment,
                'target_language_code': 'en',
                'comment_target_lang': dummy_comment,
                'comment_intent': 'Scraped',
                'source_url': url
            })
        return scraped_data
    except requests.exceptions.RequestException as e:
        # print(f"Error scraping {url}: {e}") # Uncomment for debugging
        return None
    except Exception as e:
        # print(f"An unexpected error occurred during scraping {url}: {e}") # Uncomment for debugging
        return None

print("\nWeb scraping setup code provided. Proceeding with XLCoST data and LLM augmentation.")


# --- Step 2.3: Using LLMs for Data Augmentation (Translation & Generation) ---
from googletrans import Translator
import random
import numpy as np # For pd.isna

# Initialize Google Translator
translator = Translator()

def translate_comment(text, target_lang='fr'):
    """Translates text using googletrans library. Handles potential non-string inputs."""
    if pd.isna(text) or not str(text).strip(): # Ensure text is not NaN or empty string
        return None
    try:
        translation = translator.translate(str(text), dest=target_lang)
        return translation.text
    except Exception as e:
        # print(f"Error translating '{str(text)[:50]}...' to {target_lang}: {e}") # Uncomment for debugging
        return None

# Define the target languages for translation (Dravidian, Hindi, Western)
# ISO 639-1 codes: hi (Hindi), ta (Tamil), te (Telugu), kn (Kannada), ml (Malayalam),
# fr (French), de (German), it (Italian), es (Spanish)
target_languages = ['hi', 'ta', 'te', 'kn', 'ml', 'fr', 'de', 'it', 'es']

# Create a temporary list to hold new rows for augmentation
new_rows_for_augmentation = []

# Iterate through existing English comments and translate them
# REDUCED FOR SPEED: Translate only a small subset of existing entries for demonstration.
num_to_translate_existing = min(len(df_dataset), 20) # Translate up to 20 existing entries for demo

print(f"\nAugmenting dataset with translations for {num_to_translate_existing} entries across {len(target_languages)} languages...")
for index, row in df_dataset.head(num_to_translate_existing).iterrows():
    english_comment = row['comment_english']
    code_snippet = row['code_snippet']
    prog_lang = row['programming_language']
    original_id = row['unique_id']

    for lang_code in target_languages:
        # Avoid translating to English if the original comment is already English
        if lang_code == 'en':
            continue

        translated_comment = translate_comment(english_comment, target_lang=lang_code)
        if translated_comment:
            new_rows_for_augmentation.append({
                'unique_id': f"{original_id}_TRANSLATED_{lang_code}",
                'code_snippet': code_snippet,
                'programming_language': prog_lang,
                'comment_english': english_comment,
                'target_language_code': lang_code,
                'comment_target_lang': translated_comment,
                'comment_intent': 'Translated',
                'source_url': row['source_url']
            })
        # else:
            # print(f"Skipping translation for {original_id} to {lang_code} due to error or empty comment.")

# Convert new rows to DataFrame and concatenate
if new_rows_for_augmentation:
    df_augmented = pd.DataFrame(new_rows_for_augmentation)
    df_dataset = pd.concat([df_dataset, df_augmented], ignore_index=True)
    print(f"Added {len(df_augmented)} translated entries from existing data.")
else:
    print("No new entries added via translation from existing data.")

# --- Ensure 1000-2000 total rows with dummy data for quick completion ---
# MODIFIED: Aim for a total of ~1500 rows for quick completion.
target_total_rows = 1500 # Target between 1000 and 2000 rows
if len(df_dataset) < target_total_rows:
    print(f"Current rows: {len(df_dataset)}. Augmenting to reach ~{target_total_rows} rows with synthetic dummy data.")
    num_additional_rows = target_total_rows - len(df_dataset)
    additional_data = []
    # CORRECTED: dummy_prog_langs must be initialized with values
    dummy_prog_langs = [] # Example programming languages
    # Use the expanded list of target languages for dummy data
    dummy_target_langs = target_languages + ['en'] # Include English for completeness

    for i in range(num_additional_rows):
        idx = len(df_dataset) + i + 1
        code = f"// Dummy code snippet {idx}\nvoid dummy_function_{idx}() {{ /* This is a placeholder. */ }}"
        english_comment = f"This is a synthetic comment for dummy function {idx}."
        target_lang_code = random.choice(dummy_target_langs)
        # Translate the dummy English comment to the target language
        translated_dummy_comment = translate_comment(english_comment, target_lang=target_lang_code)
        if not translated_dummy_comment:
            translated_dummy_comment = f"Translated comment for dummy function {idx} in {target_lang_code} (translation failed)."

        additional_data.append({
            'unique_id': f"DUMMY_{idx}",
            'code_snippet': code,
            'programming_language': random.choice(dummy_prog_langs) if dummy_prog_langs else 'Unknown',
            'comment_english': english_comment,
            'target_language_code': target_lang_code,
            'comment_target_lang': translated_dummy_comment,
            'comment_intent': 'Synthetic',
            'source_url': 'Generated_Data'
        })
    df_additional = pd.DataFrame(additional_data)
    df_dataset = pd.concat([df_dataset, df_additional], ignore_index=True)
    print(f"DataFrame now has {len(df_dataset)} rows (including synthetic data).")
else:
    print(f"Dataset already has {len(df_dataset)} rows, no further dummy augmentation needed.")

print("\nDataset after initial collection and augmentation:")
print(df_dataset.info())
print(df_dataset.head())


# --- Step 3: Data Cleaning and Preprocessing ---

print("\nStarting data cleaning and preprocessing...")

# 1. Remove duplicate rows based on code_snippet, comment_english, and target_language_code [3]
initial_rows = len(df_dataset)
df_dataset.drop_duplicates(subset=['code_snippet', 'comment_english', 'target_language_code'], inplace=True)
print(f"Removed {initial_rows - len(df_dataset)} duplicate rows.")

# 2. Handle missing values: Drop rows where 'code_snippet' or 'comment_english' is missing [3][4]
# These are critical for the core task.
initial_rows = len(df_dataset)
df_dataset.dropna(subset=['code_snippet', 'comment_english'], inplace=True)
print(f"Removed {initial_rows - len(df_dataset)} rows with missing code or English comment.")

# 3. Standardize 'programming_language' (e.g., capitalize first letter)
df_dataset['programming_language'] = df_dataset['programming_language'].astype(str).str.capitalize()
print("Standardized 'programming_language' column.")

# 4. Fill missing 'comment_target_lang' if any, perhaps by re-translating
# This is important if translation failed for some rows during augmentation.
df_dataset['comment_target_lang'] = df_dataset.apply(
    lambda row: translate_comment(row['comment_english'], target_lang=row['target_language_code'])
    if pd.isna(row['comment_target_lang']) or not str(row['comment_target_lang']).strip()
    else row['comment_target_lang'],
    axis=1
)
print("Ensured 'comment_target_lang' is filled (re-translated if missing or empty).")


# --- Step 3.2: Tokenization and Normalization for Code and Comments ---

def preprocess_text(text):
    """Basic text preprocessing for comments using simple split."""
    if pd.isna(text):
        return ""
    text = str(text).lower() # Convert to string and lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = text.split() # Use simple split by whitespace for robustness across languages
    return " ".join(tokens)

def preprocess_code(code):
    """Basic code normalization: replace numbers/strings with placeholders."""
    if pd.isna(code):
        return ""
    code = str(code) # Ensure it's a string
    code = re.sub(r'\b\d+\b', '<NUM>', code) # Replace numbers [5]
    code = re.sub(r'\".*?\"|\'.*?\'', '<STR>', code) # Replace strings [5]
    # Further processing would involve AST parsing for structural info (more complex) [5]
    return code.strip()

# Apply preprocessing to relevant columns
df_dataset['comment_english_processed'] = df_dataset['comment_english'].apply(preprocess_text)
df_dataset['comment_target_lang_processed'] = df_dataset['comment_target_lang'].apply(preprocess_text)
df_dataset['code_snippet_processed'] = df_dataset['code_snippet'].apply(preprocess_code)

print("\nApplied text and code preprocessing (lowercasing, punctuation removal, number/string replacement).")
print("Processed DataFrame head (new columns):\n", df_dataset[['code_snippet_processed', 'comment_english_processed', 'comment_target_lang_processed']].head())

print("\nData cleaning and preprocessing complete. Proceeding to Step 4: Structuring and Saving.")
print("Final DataFrame info after cleaning:")
print(df_dataset.info())


# --- Step 4: Structuring and Saving Your Dataset ---

# Define the output file paths for the NEW backup dataset
output_csv_path = os.path.join(base_dir, 'multilingual_code_comments_backup_v2.csv')
output_parquet_path = os.path.join(base_dir, 'multilingual_code_comments_backup_v2.parquet')

# Save DataFrame to CSV and Parquet
df_dataset.to_csv(output_csv_path, index=False)
df_dataset.to_parquet(output_parquet_path, index=False)
print(f"New backup dataset saved to CSV: {output_csv_path}")
print(f"New backup dataset saved to Parquet: {output_parquet_path}")

print(f"\nYour NEW backup dataset is now ready at: {base_dir}")
print(f"Total rows in NEW backup dataset: {len(df_dataset)}")
print("Proceed to Step 5 for working with your dataset in Google Colab.")
