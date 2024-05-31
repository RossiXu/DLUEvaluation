import json
import re
import os
import pandas as pd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    SOFT_GREEN = '\033[38;5;77m'  # 选择了一个较为柔和的绿色
    SOFT_RED = '\033[38;5;124m'  # 选择了一个较为柔和的红色


def read_file(file_name, split_str=None):
    if 'jsonl' in file_name:
        datas = []
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                datas.append(data)
        return datas
    elif 'json' in file_name:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif 'xlsx' in file_name:
        return pd.read_excel(file_name)
    elif 'csv' in file_name:
        return pd.read_csv(file_name)
    else:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.read()
        if split_str:
            elements = data.split(split_str)
            elements = [e.strip() for e in elements if e.strip()]
            return elements
        else:
            return data


def write_file(file_name, data, split_str=None):
    if type(data) is list:
        lists = data
        if 'jsonl' in file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                for element in lists:
                    json.dump(element, f)
                    f.write('\n')
        else:
            split_str = '\n' if not split_str else split_str
            with open(file_name, 'w', encoding='utf-8') as f:
                for element in lists:
                    f.write(str(element))
                    f.write(split_str)
    elif type(data) is dict:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    elif type(data) is pd.DataFrame:
        if 'xlsx' in file_name:
            data.to_excel(file_name, index=False)
        elif 'csv' in file_name:
            data.to_csv(file_name, index=False)
        else:
            print(f'Cannot save {type(data)} to {file_name}.')
    else:
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(str(data))


def walk(path, full_path=True):
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if full_path:
                filename = os.path.join(root, file)
            else:
                filename = file
            filenames.append(filename)
    return filenames


def get_llm_series_name(llm_name):
    if llm_name in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'davinci', 'text-davinci-001', 'text-davinci-002', 'text-davinci-003']:
        return 'GPT'
    elif llm_name in ['vicuna-7b', 'vicuna-13b', 'vicuna-33b']:
        return 'Vicuna'
    elif llm_name in ['chatglm2-6b']:
        return 'ChatGLM'
    elif llm_name in ['llama-2-7b-chat', 'llama2-13b-chat', 'llama-2-70b-chat']:
        return 'LLaMA'
    elif 'mistral' in llm_name or 'mixtral' in llm_name:
        return 'Mixtral'

    return 'None'


def merge_columns(old_data_path, new_data_path, columns=None, key='ID', save_path=None, output=False):
    old_data = read_file(old_data_path) if type(old_data_path) == str else old_data_path
    new_data = read_file(new_data_path) if type(new_data_path) == str else new_data_path
    columns = columns if columns else [c for c in new_data.columns if c not in old_data.columns]
    for column in columns:
        if column not in list(old_data.columns):
            old_data.loc[column] = pd.Series().copy()

    for index, row in old_data.iterrows():
        new_row = new_data[new_data[key] == row[key]]
        if new_row.shape[0] == 1:
            for column in columns:
                old_data.loc[index, column] = new_row.iloc[0][column]

    if save_path:
        write_file(save_path, old_data)
    if output:
        print("Update Columns %s!" % ('&'.join(columns)))
    return old_data


def extract_answer(text):
    answer = 'None'
    if type(text) is str:
        text = text.strip()
        # check: A.
        r = re.search(r'([ABCDEFG])\.', text)
        if r:
            return r.group(1)
        # check A)
        r = re.search(r'([ABCDEFG])\)', text)
        if r:
            return r.group(1)
        # check: A:
        r = re.search(r'([ABCDEFG]):', text)
        if r:
            return r.group(1)
        # check A
        r = re.search(r'([ABCDEFG])', text)
        if r:
            return r.group(1)
    return answer
