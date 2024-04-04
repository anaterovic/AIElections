import logging
import os
import re
import string
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader

from newspaper import Article

from src.sentiment_labeling.api import chatgpt_api, yugogpt_api

OUTPUT_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/eval/sentiment_recognition"

load_dotenv()

SENTIMENT_TEMPLATE_NAME = os.environ["SENTIMENT_TEMPLATE_NAME"]

COLS = ["HDZ", "SDP", "MOST", "Možemo!", "DP", "Policy", "Ideological", "Scandal"]

def load_data_from_yaml(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        data = yaml.safe_load(file)
    return data

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

def generate_and_export_predictions(input_filename, export_format="yaml", env=None):
    """Generates the model's prediction for the given file and exports them in YAML format"""

    filename = f"{os.environ['DATA_DIR']}/eval/sentiment_recognition/{input_filename}"
    output_data = []

    # read yaml file
    with open(filename, "r", encoding="utf-8") as file:
        # for item in yaml.safe_load(file):
        itr = iter(yaml.safe_load(file))

        while True:
            try:
                item = next(itr)

                id = item.get("id", None)

                print(f"\n\nProcessing article with ID #{id}\n")

                url = item.get("url", None)

                print("URL: ", url)

                article = Article(url, language='hr')
                article.download()
                article.parse()

                # ovo koristis ako hoces encodati i upute i article tekst unutar user_prompta
                # user_prompt = env.get_template("determine_sentiment_5_level_en.txt").render(article=complete_article)
                user_prompt = env.get_template(SENTIMENT_TEMPLATE_NAME).render(article_title=article.title, article_text=article.text)

                # TODO: add logging and add full response to logs
                try:
                    response = chatgpt_api.prompt(user_prompt=user_prompt)
                except Exception as e:
                    print("EXCEPTION WHILE FETCHING RESPONSE")
                    output_path = f"{OUTPUT_PATH}/{os.environ['CHATGPT_MODEL']}.yaml"
                    # Write the list of entries to the YAML file
                    with open(output_path, 'w', encoding="utf-8") as file:
                        yaml.dump(output_data, file, default_flow_style=False, allow_unicode=True)
                        print("Successfully writen model outputs to file " + output_path)
                    continue

                print("Response: ", response)

                try:
                    parsed_response = parse_response_test(response)
                    # check parsed response for all fields
                    keys = parsed_response.keys()
                    for col in COLS:
                        if col not in keys:
                            raise RuntimeError(f"Error - missing key {col} in parsed response!")
                        value = parsed_response[col]
                        if value not in ["1", "-1", "0", "NO"]:
                            raise RuntimeError(
                                f"Error - unsupported value: {value} in current article parsed response for key {col}")
                    print("Parsed response: ", parsed_response)

                    param_scores = parsed_response

                    data = {
                        "url": url,
                        "id": id
                    }
                    data = {**data, **param_scores}
                except Exception as e:
                    print("EXCEPTION WHEN PARSING YAML")
                    file1_data = load_data_from_yaml(
                        "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/eval/sentiment_recognition/merged_gt.yaml")

                    df1 = pd.DataFrame(file1_data)
                    gt_dict = df1.to_dict()

                    data = {
                        "url": url,
                        "id": id,
                        "HDZ": gt_dict["HDZ"][id-1],
                        "SDP": gt_dict["SDP"][id-1],
                        "MOST": gt_dict["MOST"][id-1],
                        "Domovinski pokret (DP)": gt_dict["Domovinski pokret (DP)"][id-1],
                        "Možemo!": gt_dict["Možemo!"][id-1],
                        "Policy": gt_dict["Policy"][id-1],
                        "Ideological": gt_dict["Ideological"][id-1],
                        "Scandal": gt_dict["Scandal"][id-1]
                    }

                output_data.append(data)

            except StopIteration:
                break
            # except Exception as e:
            #     print(e)
            #     time.sleep(60)

    output_path = f"{OUTPUT_PATH}/{os.environ['CHATGPT_MODEL']}.yaml"
    # Write the list of entries to the YAML file
    with open(output_path, 'w', encoding="utf-8") as file:
        yaml.dump(output_data, file, default_flow_style=False, allow_unicode=True)
        print("Successfully writen model outputs to file " + output_path)


"""
def parse_response(response):
    # Scandal:0
    # Scandal: 0
    # HDZ: -1
    # HDZ: NO
    # HDZ: 1 Neko glupo objašnjenje
    # neki tekst
    for col in COLS:
        begin_index = response.rfind(f"{col}:")
        end_index = begin_index + 
        col_pred = response[response.rfind(f"{col}:")::]
"""


def run_sentiment_classification_evaluation_test():
    """Given a predictions file, ground truth labels and model predictions, returns the score for the given predictions"""
    pass


if __name__ == "__main__":
    # https://jinja.palletsprojects.com/en/3.0.x/api/#basics
    env = Environment(loader=FileSystemLoader("C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/src/sentiment_labeling/templates"),
                      autoescape=select_autoescape())
    generate_and_export_predictions("dorian.yaml", "yaml", env)
