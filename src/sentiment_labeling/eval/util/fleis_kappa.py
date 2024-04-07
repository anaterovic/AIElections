import pandas as pd
import yaml
from collections import defaultdict
import numpy as np
from itertools import groupby
from collections import Counter
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
def load_data_from_yaml(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        data = yaml.safe_load(file)
    return data

def prepare_data(data):
    prepared_data = defaultdict(list)
    for entry in data:
        for key, value in entry.items():
            if key != 'id' and key != 'url':
                prepared_data[key].append(value)
    return prepared_data

def match_rows(file1_data, file2_data):
    matched_data = defaultdict(list)
    for id, entries in file1_data.items():
        if id in file2_data:
            matched_data[id] = (entries, file2_data[id])
    return matched_data

if __name__ == "__main__":
    # Load data from YAML filesd
    file1_data = load_data_from_yaml(
        "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/eval/sentiment_recognition/merged_gt.yaml")
    file2_data = load_data_from_yaml(
        "C:/Users/dsmoljan/Desktop/AI izbori/parlametrika/data/eval/sentiment_recognition/GPT-3.5.yaml")

    df1 = pd.DataFrame(file1_data)
    df2 = pd.DataFrame(file2_data)

    dict_mappings = {"-1": 1, "0": 1, "1": 2, "False": 3}

    for col in df1.columns:
        if col in ["id", "url"]:
            continue
        col_values_1 = []
        col_values_2 = []
        for v in df1[[col]].values:
            value = v[0]
            if value != "False":
                value = int(value)
            col_values_1.append(dict_mappings[str(value)])
            #col_values_1.append(str(v[0]))

        for v in df2[[col]].values:
            value = v[0]
            if value != "False":
                try:
                    value = int(value)
                except Exception as e:
                    pass
            col_values_2.append(dict_mappings[str(value)])
            #col_values_2.append(str(v[0]))

        # https://stackoverflow.com/questions/51919897/is-fleiss-kappa-a-reliable-measure-for-interannotator-agreement-the-following-r
        stacked_col_values = np.transpose(np.vstack((col_values_1, col_values_2)))
        prepared_aggr_data = aggregate_raters(stacked_col_values)
        print(f"Column: {col}, fleiss kappa: {fleiss_kappa(prepared_aggr_data[0], method='fleiss')}")


    # Prepare data
    # file1_prepared_data = prepare_data(file1_data)
    # file2_prepared_data = prepare_data(file2_data)

    # Match rows based on ID
    # matched_data = match_rows(file1_prepared_data, file2_prepared_data)
    #
    # # Calculate Fleiss's coefficient for each category
    # coefficients = {}
    # for category, (file1_values, file2_values) in matched_data.items():
    #     data = [file1_values, file2_values]
    #     kappa = fleiss_kappa(data)
    #     coefficients[category] = kappa
    #
    # # Output the coefficients
    # for category, kappa in coefficients.items():
    #     print(f"Fleiss's coefficient for category '{category}': {kappa}")