import re
import json
from sklearn.model_selection import train_test_split


def clean_data(sent):
    sent = sent.replace('\n', ' ').replace('\\n', ' ').replace('\\', ' ').replace("\"", ' ')
    sent = re.sub('<[^<]+?>', '', sent)
    return sent


def nli_dataset(data_path, dataset):
    for name in ('train', 'test', 'dev'):
        res = []
        with open(data_path + '/' + name + '.doc.json', 'r', encoding='utf-8') as f:
            docs = json.load(f)
        with open(data_path + '/' + name + '.ann.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)
        for doc in docs:
            doc_id = doc['id']
            hypothesis = clean_data(doc['hypothesis'])
            premise = clean_data(doc['premise'])
            label = labels[doc_id]['label']
            dic = {'hypothesis': hypothesis, 'premise': premise, 'label': label}
            res.append(dic)
        with open(f'/shared_home/guanxinyan2022/clean_data/{dataset}/{name}.json', 'w') as f:
            json.dump(res, f, indent=2)
        print(f'{dataset} {name} {len(res)}')
    return


def hyper_dataset(data_path, dataset):
    # 先把train分割为train和dev

    with open(data_path + '/train.doc.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)
    with open(data_path + '/train.ann.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)

    res = []
    for doc_id in range(len(docs)):
        doc = clean_data(docs[doc_id]['document'])
        label = labels[doc_id]['label']
        dic = {'doc': doc, 'label': label}
        res.append(dic)

    data_train, data_dev = train_test_split(res, train_size=0.8, random_state=42)
    with open(f'/shared_home/guanxinyan2022/clean_data/{dataset}/train.json', 'w') as f:
        json.dump(data_train, f, indent=2)

    with open(f'/shared_home/guanxinyan2022/clean_data/{dataset}/dev.json', 'w') as f:
        json.dump(data_dev, f, indent=2)

    # 读取test
    with open(data_path + '/test.doc.json', 'r', encoding='utf-8') as f:
        docs = json.load(f)
    with open(data_path + '/test.ann.json', 'r', encoding='utf-8') as f:
        labels = json.load(f)

    res = []
    for doc_id in range(len(docs)):
        doc = clean_data(docs[doc_id]['document'])
        label = labels[doc_id]['label']
        dic = {'doc': doc, 'label': label}
        res.append(dic)

    with open(f'/shared_home/guanxinyan2022/clean_data/{dataset}/test.json', 'w') as f:
        json.dump(res, f, indent=2)

    print(f'Hyper train {len(data_train)}\n Hyper dev {len(data_dev)} \n Hyper test {len(res)}')


nli_dataset('/shared_home/guanxinyan2022/data/ContractNLI', 'ContractNLI')
nli_dataset('/shared_home/guanxinyan2022/data/DocNLI', 'DocNLI')
hyper_dataset('/shared_home/guanxinyan2022/data/Hyperpartisan', 'Hyperpartisan')
