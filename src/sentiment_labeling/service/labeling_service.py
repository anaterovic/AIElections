import os
import re

from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from newspaper import Article

from src.sentiment_labeling.api import chatgpt_api, yugogpt_api

ARTICLES_LISTS_PATH = "data/eval/sentiment_recognition/article_list"
OUTPUT_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/eval/sentiment_recognition"
CHECKPOINT_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/labeling/progress/checkpoint.yaml"
START_DATE = "01.06.2020" # or whatever is the first date in the articles list TODO: print a warning if the first date in the articles list is after this
END_DATE = "01.04.2024" # or whatever is the oldest date in the articles list TODO: print a warning if the last date inn the articles list is before this

load_dotenv()

SENTIMENT_TEMPLATE_NAME = os.environ["SENTIMENT_TEMPLATE_NAME"]

COLS = ["HDZ", "SDP", "MOST", "Možemo!", "DP", "Policy", "Ideological", "Scandal"]
OUTLETS = ["24sata.hr", "dnevnik.hr", "jutarnji.hr", "vecernji.hr", "index.hr"]

def load_data_from_yaml(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        data = yaml.safe_load(file)
    return data

# ideja
# skinuti popis članaka po svakom portalu -> čak je možda OK da imamo jedan zajednički file pa filtriramo po portalu
# Index, Dnevnik, Jutarnji, Večernji, 24sata
# iterirati po svakom od fileova
# imati poseban file progress.txt gdje se zapisuje progress u idućem formatu
# portals_processed: X,Y,Z
# portal_x_date_from: najraniji_datum_u_fileu_za
# portal_x_date_to: datum do kojeg je došlo procesiranje članaka za određeni portal
# prilikom SVAKE grepške, paljenja i gašenja programa ažurira se ovaj file, tako da se može lako nastaviti ako se program ugasi
def fetch_article_sentiment(article: Article):
    """Fetches sentiment for given article and returns it as a dict"""
    article.download()
    article.parse()

    user_prompt = env.get_template(SENTIMENT_TEMPLATE_NAME).render(article_title=article.title,
                                                                   article_text=article.text)

    # TODO: add logging and add full response to logs
    try:
        response = chatgpt_api.prompt(user_prompt=user_prompt)
    except Exception as e:
        print("EXCEPTION WHILE FETCHING RESPONSE")
        # TODO: zapisi ovaj entry u odgovarajuci CSV file kako bi se mogao kasnije obraditi!!!
        return None

    print("Response: ", response)

    parsed_response = parse_response_test(response)
    # check parsed response for all fields
    keys = parsed_response.keys()
    for col in COLS:
        if col not in keys:
            raise RuntimeError(f"Error - missing key {col} in parsed response!")
        value = parsed_response[col]
        if value not in ["1", "-1", "0", "NO"]:
            raise RuntimeError(f"Error - unsupported value: {value} in current article parsed response for key {col}")
    print("Parsed response: ", parsed_response)

    pass


def generate_and_export_predictions(input_filename, export_format="yaml", env=None):
    """Generates the model's prediction for the given file and exports them in YAML format"""

    filename = f"{os.environ['DATA_DIR']}/labeling/article_list/{input_filename}"
    output_data = []

    df = pd.read_csv(filename)

    checkpoint_data = load_data_from_yaml(CHECKPOINT_PATH)

    for outlet in OUTLETS:
        start_date = START_DATE
        end_date = END_DATE
        # check if outlet has already been fully processed
        if checkpoint_data[outlet]["finished"]:
            continue
        # check if outlet has been partially processed; if it has; continue processing from the last processed date
        if checkpoint_data[outlet]["latest_processed_date"] != None:
            start_date = checkpoint_data[outlet]["latest_processed_date"]
        outlet_df = df[df['outlet'] == outlet]
        # Iterate over rows
        for index, row in tqdm(outlet_df.iterrows(), total=len(outlet_df), desc=f'Processing {outlet}'):
            # row['outlet'], row['id'], row['url'], row['title'], row['date_published']
            id = row['id']
            url = row['url']
            title = row['title']
            date_published = row['date_published']

            article = Article(url, language='hr')
            fetch_article_sentiment(article)

            pass
    pass

def parse_response(response):
    """Parses ChatGPT's response to try and fetch sentiment on all possible parties"""
    pass

def normalize_party_name(party_name):
    # Define mappings for variations of party names
    mappings = {
        "Možemo!": "Možemo!",
        "Domovinski pokret": "DP",
        "Domovinski pokret (DP)": "DP",
        "DP": "DP",
        "Možemo": "Možemo!"
    }
    # Return the normalized name if mapping exists, otherwise return the original name
    return mappings.get(party_name, party_name)


def parse_response_test(response_text):
    # Define regular expressions for extracting sentiments and reasoning
    sentiment_pattern = r"(HDZ|SDP|MOST|DP|Možemo(?:!)?|Domovinski\s+pokret\s*(?:\(DP\))?|Policy|Ideological|Scandal): (-?\d+|NO)"
    reasoning_pattern = r"Reasoning: (.+?)###"

    # Search for sentiments and reasoning using regular expressions
    sentiments = {}
    for match in re.finditer(sentiment_pattern, response_text):
        party_name, sentiment = match.group(1), match.group(2)
        normalized_party_name = normalize_party_name(party_name)
        sentiments[normalized_party_name] = sentiment

    reasoning_match = re.search(reasoning_pattern, response_text)

    # Extract reasoning if found
    reasoning = reasoning_match.group(1) if reasoning_match else None

    return sentiments

if __name__ == "__main__":
    # https://jinja.palletsprojects.com/en/3.0.x/api/#basics
    env = Environment(loader=FileSystemLoader("C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/src/sentiment_labeling/templates"),
                      autoescape=select_autoescape())
    generate_and_export_predictions("articles.csv", "yaml", env)


