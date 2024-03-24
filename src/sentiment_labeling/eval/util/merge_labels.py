import os

import yaml
from collections import Counter
from pathlib import Path

LABEL_DIR_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/eval/sentiment_recognition"
OUTPUT_PATH = "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/eval/sentiment_recognition"

default_cols = ['id', 'url']
cols = ['HDZ', 'SDP', 'Možemo!', 'MOST', 'Domovinski pokret (DP)', 'Policy', 'Ideological', 'Scandal']

def load_data_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

def majority_voting(data_list):
    output_data = []
    for i in range(len(data_list[0])): # za svaki novinski članak u yamlicama
        id = data_list[0][i]['id']
        url = data_list[0][i]['url']

        news_article_data = {
            "url": url,
            "id": id,
        }

        # calculate majority vote for each party
        for col in cols:
            party_labels = []
            for j in range(len(data_list)):
                party_labels.append(data_list[j][i][col])

            majority_party_label = None
            party_labels_amounts = Counter(party_labels)
            max_count = max(party_labels_amounts.values())
            for party_label in party_labels_amounts.keys():
                if (party_labels_amounts[party_label] == max_count):
                    if majority_party_label is None:
                        majority_party_label = party_label
                    else:
                        # if you've already found a party label with the same number of votes
                        # decide on the neutral label if possible (labeled as False in yml), if not, keep the first label found
                        if party_label == False:
                            majority_party_label = False

            news_article_data[col] = majority_party_label

        output_data.append(news_article_data)

    return output_data

def write_merged_data_to_yaml(merged_data, output_file_path):
    with open(output_file_path, 'w') as file:
        yaml.dump(merged_data, file)

if __name__ == "__main__":
    directory_path = Path(LABEL_DIR_PATH)
    files = [os.path.join(LABEL_DIR_PATH,file.name) for file in directory_path.iterdir() if file.is_file()]
    print("Files in directory:", files)

    data_list = []

    for file_path in files:
        data = load_data_from_yaml(file_path)
        data_list.append(data)

    pass
    # perform majority voting
    merged_data = majority_voting(data_list)

    # write merged data to YAML file
    output_path = f"{OUTPUT_PATH}/merged_gt.yaml"
    # Write the list of entries to the YAML file
    with open(output_path, 'w', encoding="utf-8") as file:
        yaml.dump(merged_data, file, default_flow_style=False, allow_unicode=True)
        print("Successfully writen merged labels to file " + output_path)