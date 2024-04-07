import ast
import logging
import os
import re

import tiktoken
from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader

from newspaper import Article

from src.sentiment_labeling.api import chatgpt_api, yugogpt_api

ARTICLES_LISTS_PATH = "data/eval/sentiment_recognition/article_list"
OUTPUT_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/labeling/output"
CHECKPOINT_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/labeling/progress"
START_DATE = "01.06.2020" # or whatever is the first date in the articles list TODO: print a warning if the first date in the articles list is after this
END_DATE = "01.04.2024" # or whatever is the oldest date in the articles list TODO: print a warning if the last date inn the articles list is before this

LOG_DIR = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/logs/labeling"

load_dotenv()

SENTIMENT_TEMPLATE_NAME = os.environ["SENTIMENT_TEMPLATE_NAME"]
INPUT_TOKEN_PRICE_PER_1K = float(os.environ["INPUT_TOKEN_PRICE_PER_1K"])
OUTPUT_TOKEN_PRICE_PER_1K = float(os.environ["OUTPUT_TOKEN_PRICE_PER_1K"])

COLS = ["HDZ", "SDP", "MOST", "Možemo!", "DP", "Policy", "Ideological", "Scandal", "Impact"]
OUTLETS = ["24sata.hr", "dnevnik.hr", "jutarnji.hr", "vecernji.hr", "index.hr"]


def setup_logger(log_file):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create a file handler
    file_handler = logging.FileHandler(log_file)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def load_data_from_yaml(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        data = yaml.safe_load(file)
    return data

def log_or_print(text, logger, msg_type="info"):
    if logger is None:
        print(text)
    else:
        if msg_type == "info":
            logger.info(text)
        elif msg_type == "error":
            logger.error(text)
        else:
            logger.warn(text)

# ideja
# skinuti popis članaka po svakom portalu -> čak je možda OK da imamo jedan zajednički file pa filtriramo po portalu
# Index, Dnevnik, Jutarnji, Večernji, 24sata
# iterirati po svakom od fileova
# imati poseban file progress.txt gdje se zapisuje progress u idućem formatu
# portals_processed: X,Y,Z
# portal_x_date_from: najraniji_datum_u_fileu_za
# portal_x_date_to: datum do kojeg je došlo procesiranje članaka za određeni portal
# prilikom SVAKE grepške, paljenja i gašenja programa ažurira se ovaj file, tako da se može lako nastaviti ako se program ugasi
def fetch_article_sentiment(article: Article, newslines = [], logger = None):
    """Fetches sentiment for given article and returns it as a dict"""
    article.download()
    article.parse()

    mentioned_parties = ','.join(newslines)

    user_prompt = env.get_template(SENTIMENT_TEMPLATE_NAME).render(article_title=article.title,
                                                                   article_text=article.text,
                                                                   mentioned_parties=mentioned_parties)

    # TODO: add logging and add full response to logs

    response = chatgpt_api.prompt(user_prompt=user_prompt)

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    log_or_print("Prompt: " + user_prompt, logger)
    log_or_print("Response: " + response, logger)

    no_request_tokens = len(encoding.encode(user_prompt))
    no_response_tokens = len(encoding.encode(response))

    request_price = (no_request_tokens / 1000) * INPUT_TOKEN_PRICE_PER_1K + (no_response_tokens / 1000) * OUTPUT_TOKEN_PRICE_PER_1K

    parsed_response = parse_response_test(response)
    # check parsed response for all fields
    keys = parsed_response.keys()
    for col in COLS:
        if col not in keys:
            raise RuntimeError(f"Error - missing key {col} in parsed response!")
        value = parsed_response[col]
        if (col != "Impact" and value not in ["1", "-1", "0", "NO"]) or (col == "Impact" and value not in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]):
            raise RuntimeError(f"Error - unsupported value: {value} in current article parsed response for key {col}")
    log_or_print("Parsed response: " + str(parsed_response), logger)
    return parsed_response, request_price

def generate_and_export_predictions(input_filename, outlet, export_format="yaml", env=None):
    """Generates the model's prediction for the given file and exports them in YAML format"""

    outlet_name = outlet.split(".hr")[0]
    logger = setup_logger(LOG_DIR + f"/{outlet_name}.txt")
    filename = f"{os.environ['DATA_DIR']}/labeling/article_list/{input_filename}"
    output_df = pd.DataFrame(columns=["id", "url", "title", "date", *COLS])
    skipped_articles = []
    output_data = []

    checkpoint_path = os.path.join(CHECKPOINT_PATH, outlet_name + ".yaml")

    df = pd.read_csv(filename)

    checkpoint_data = load_data_from_yaml(checkpoint_path)

    start_date = START_DATE
    end_date = END_DATE
    # check if outlet has already been fully processed
    if checkpoint_data["finished"] == "True":
        return
    # check if outlet has been partially processed; if it has; continue processing from the last processed date
    if checkpoint_data["latest_processed_date"] != "None":
        start_date = checkpoint_data["latest_processed_date"]
        output_df = pd.read_csv(os.path.join(OUTPUT_PATH, outlet_name + ".csv"))

    outlet_df = df[df['outlet'] == outlet]
    outlet_df = outlet_df[df['date_published'] >= start_date]
    outlet_df = outlet_df.drop_duplicates(subset='title')
    # Iterate over rows
    counter = 0
    skipped = 0
    total_price = 0
    progress_bar = tqdm(total=len(outlet_df),
                        desc=f'Processing {outlet} | Skipped: {skipped} | Total Price: {total_price:.2f}')

    current_date = start_date

    for index, row in outlet_df.iterrows():
        # row['outlet'], row['id'], row['url'], row['title'], row['date_published']
        if counter % 5 == 0:
            # TODO: dodaj da se radi export za trenutni outlet nakon npr. svakih 100 članaka
            # i usput se ažurira checkpoint_outlet.yaml, također se ažurira lista skippanih članaka
            #output_df = pd.merge(pd.DataFrame(output_data), output_df, on="id", how="inner")
            output_df = pd.concat([pd.DataFrame(output_data), output_df], ignore_index=True)
            output_df = output_df.drop_duplicates(subset='id')
            output_df.to_csv(os.path.join(OUTPUT_PATH, outlet_name + ".csv"))
            output_data = []

            checkpoint_info = {"finished": "False","latest_processed_date": current_date}

            with open(checkpoint_path, 'w', encoding="utf-8") as file:
                yaml.dump(checkpoint_info, file, default_flow_style=False, allow_unicode=True)

            skipped_df = pd.DataFrame(skipped_articles)
            skipped_df.to_csv(os.path.join(OUTPUT_PATH, outlet_name + "-skipped.csv"))

            logger.info("Dumping currently processed files and checkpoint")



        id = row['id']
        url = row['url']
        title = row['title']
        date_published = row['date_published']
        current_date = date_published
        newslines = ast.literal_eval(row['newslines'])

        article = Article(url, language='hr')

        request_price = 0

        try:
            article_labels, request_price = fetch_article_sentiment(article, newslines, logger)
            article_labels.update({"id": id, "url": url, "title": title, "date": date_published})
            output_data.append(article_labels)
        except Exception as e:
            # if there is an exception while fetching the article info
            # save the article to a special file and continue processing other articles
            log_or_print(f"EXCEPTION WHILE FETCHING RESPONSE FOR ARTICLE: {title}\n {e}", logger, "error")
            skipped += 1
            skipped_articles.append(dict(row))

        total_price += request_price
        counter += 1

        # Update the progress bar description
        progress_bar.set_description(f'Processing {outlet} | Skipped: {skipped} | Total Price: {total_price:.5f}$')
        progress_bar.update(1)

    output_df = pd.concat([pd.DataFrame(output_data), df], ignore_index=True)
    output_df.to_csv(os.path.join(OUTPUT_PATH, outlet + ".csv"))

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
    sentiment_pattern = r"(HDZ|SDP|MOST|DP|Možemo(?:!)?|Domovinski\s+pokret\s*(?:\(DP\))?|Policy|Ideological|Scandal|Impact): (-?\d+|NO)"
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
    input_outlet = None
    while input_outlet not in OUTLETS:
        input_no = input("Please select the name of the outlet:\n 1. 24sata.hr \n 2. dnevnik.hr \n 3. jutarnji.hr \n 4. vecernji.hr \n 5. index.hr\n ")

        if input_no not in ["1", "2", "3", "4", "5"]:
            print("\nError - unsupported outlet! Please select an outlet from 1-5!\n")
            continue

        index = int(input_no)-1
        input_outlet = OUTLETS[index]

    print(f"Selected outlet {input_outlet}")
    generate_and_export_predictions("articles.csv", input_outlet, "yaml", env)


