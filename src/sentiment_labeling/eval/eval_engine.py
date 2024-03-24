import logging
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import yaml
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader

from newspaper import Article

from src.sentiment_labeling.api import chatgpt_api, yugogpt_api

OUTPUT_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/DATA/eval/sentiment_recognition"

load_dotenv()

SENTIMENT_TEMPLATE_NAME = os.environ["SENTIMENT_TEMPLATE_NAME"]


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

                # print("Article publish date: ", article.publish_date)
                print("Article title: ", article.title)
                # print("Article text: ", article.text)

                complete_article = "Naslov: " + article.title + "\n" + article.text

                # ovo koristis ako hoces encodati i upute i article tekst unutar user_prompta
                # user_prompt = env.get_template("determine_sentiment_5_level_en.txt").render(article=complete_article)
                system_prompt = env.get_template(SENTIMENT_TEMPLATE_NAME).render()

                response = chatgpt_api.prompt(system_prompt=system_prompt, user_prompt=complete_article)

                print("Response: ", response)

                # param_scores = yaml.safe_load(response)
                #
                # data = {
                #     "url": url,
                #     "id": id
                # }
                #
                # data = {**data, **param_scores}
                #
                # output_data.append(data)


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


def run_sentiment_classification_evaluation_test():
    """Given a predictions file, ground truth labels and model predictions, returns the score for the given predictions"""
    pass


if __name__ == "__main__":
    # https://jinja.palletsprojects.com/en/3.0.x/api/#basics
    env = Environment(loader=FileSystemLoader("C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/src/sentiment_labeling/templates"),
                      autoescape=select_autoescape())
    generate_and_export_predictions("dorian_test.yaml", "yaml", env)
