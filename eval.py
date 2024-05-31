import ast
import time

from utils.utils import *
from utils.answer import chat_with_llm_multi_thread
from my_models import *
from utils.config import *
import random
from rouge import Rouge
import string
from collections import Counter


def read_data(data_dir='data/DocumentClassification', dataset='ContractNLI', num=100, split='test'):
    data_base = data_dir + '/' + dataset + '/'

    try:
        with open(data_base + split + '.doc.json', 'r', encoding='utf-8') as f:
            docs = json.load(f)
        with open(data_base + split + '.ann.json', 'r', encoding='utf-8') as f:
            anns = json.load(f)
    except FileNotFoundError:
        return []

    examples = []

    if data_dir == 'data/DocumentClassification':
        if dataset == 'ContractNLI':
            for doc in docs:
                doc_id = doc['id']
                premise = doc['premise']
                hypotheis = doc['hypothesis']
                label = [ann for ann in anns if ann['id'] == doc_id][0]['label']
                example = {'id': doc_id, 'document': premise, 'hypothesis': hypotheis, 'label': label}
                examples.append(example)

        if dataset == 'Hyperpartisan':
            for doc in docs:
                doc_id = doc['id']
                doc = doc['document']
                label = [ann for ann in anns if ann['id'] == doc_id][0]['label']
                example = {'id': doc_id, 'document': doc, 'label': label}
                examples.append(example)

    if data_dir == 'data/DocumentStructureAnalysis':
        is_golden_context = False
        if dataset == 'ECOM' or dataset == 'APE':
            for doc in docs:
                doc_id = doc['id']
                sents = doc['document']
                labels = [ann for ann in anns if ann['id'] == doc_id][0]['labels']
                if is_golden_context:
                    sents = [sent for idx, sent in enumerate(sents) if labels[idx] != 'O']
                    labels = [label for label in labels if label != 'O']
                example = {'id': doc_id, 'document': sents, 'label': labels}
                examples.append(example)

    if data_dir == 'data/DocumentExtraction':
        if dataset == 'LitBank' or dataset == 'NarrativeQA':
            for doc in docs:
                doc_id = doc['id']
                context = doc['document']
                question = doc['question']
                answers = [ann for ann in anns if ann['id'] == doc_id][0]['answer']
                example = {'id': doc_id, 'document': context, 'question': question, 'label': answers}
                examples.append(example)

    if data_dir == 'data/Transcription':
        if dataset == 'SummScreen' or dataset == 'GovReport':
            for doc in docs:
                doc_id = doc['id']
                document = doc['document']
                summary = [ann for ann in anns if ann['id'] == doc_id][0]['summary']
                example = {'id': doc_id, 'document': document, 'label': summary}
                examples.append(example)

        if dataset == 'Qasper':
            for doc in docs:
                doc_id = doc['id']
                document = doc['document']
                question = doc['question']
                answer = [ann for ann in anns if ann['id'] == doc_id][0]['answer']
                example = {'id': doc_id, 'document': document, 'question': question, 'label': answer}
                examples.append(example)

    if num > 0 and num <= len(examples):
        examples = random.sample(examples, num)
    elif num > 0 and num > len(examples):
        print(
            f'{bcolors.WARNING}There are only {len(examples)} examples in {dataset}, less than required num {num}!{bcolors.ENDC}')
    return examples


def construct_input(instance, dataset, max_token_len=4000, slient=True):
    if dataset == 'Hyperpartisan':
        prompts = ["For each snippet of text, label the standpoint of the text as extreme or not extreme. "
                   "The answer should be exact 'extreme' or 'not extreme'.\n\n"
                   "Text: %s\n"
                   "Standpoint:",

                   "For each snippet of text, identify whether the news takes an extreme left-wing or right-wing "
                   "standpoint. "
                   "The answer should be exact a word, 'true' or 'false', without explanation.\n\n"
                   "Text: %s\n"
                   "Label:",

                   "For each snippet of text, identify whether the news takes an extreme left-wing or right-wing "
                   "standpoint. "
                   "The answer should be exact a word, 'true' or 'false', without explanation.\n\n"
                   "Text: %s\n"
                   "Standpoint:",
                   ]
        prompt_idx = 2
        return prompts[prompt_idx] % instance['document']

    if dataset == 'ContractNLI':
        prompts = ["Please identify whether the premise entails the hypothesis. "
                   "The answer should be exact 'Entailment', 'Contradiction' or 'NotMentioned', without explanation.\n\n"
                   "Premise: %s\n"
                   "Hypothesis: %s\n"
                   "Answer:"]
        prompt_idx = 0
        hypothesis = instance['hypothesis']
        premise = ' '.join(instance['document'].split())

        max_premise_len = max_token_len - len((prompts[prompt_idx] + hypothesis).split())
        if max_premise_len < len(premise.split()):
            if not slient:
                print('Truncated %d to %d tokens.' % (len(premise.split()), max_premise_len))
            premise = ' '.join(premise.split()[:max_premise_len])

        prompt = prompts[prompt_idx] % (premise, hypothesis)
        if not slient:
            print('Input length: ', len(prompt.split()))
        return prompt

    if dataset == 'SummScreen' or dataset == 'GovReport':
        prompts = ["%s\n\n"
                   "TL;DR:",

                   "%s\n\n"
                   "Summarize the document in around 100 words."]
        prompt_idx = 0

        max_text_len = max_token_len - len(prompts[prompt_idx].split())
        text = instance['document']
        if max_text_len < len(text.split()):
            if not slient:
                print('Truncated %d to %d tokens.' % (len(text.split()), max_text_len))
            text = ' '.join(text.split()[:max_text_len])

        return prompts[prompt_idx] % text

    if dataset == 'Qasper':
        prompts = ["Please answer the given question based on the context."
                   "If the answer is not present in the context, please reply with 'unanswerable'."
                   "If the question is a closed-ended question, please only reply with 'yes' or 'no'.\n\n"
                   "Context: %s\n"
                   "Question: %s\n"
                   "Answer:",
                   ]
        prompt_idx = 0
        document = instance['document']
        question = instance['question']
        max_text_len = max_token_len - len(prompts[prompt_idx].split()) - len(question.split())
        if max_text_len < len(document.split()):
            if not slient:
                print('Truncated %d to %d tokens.' % (len(document.split()), max_text_len))
            document = ' '.join(document.split()[:max_text_len])
        return prompts[prompt_idx] % (document, question)

    if dataset == 'ECOM':
        prompts = ["Please extract opinion segments from the given text."
                   "Each opinion segment contains continuous one or more sentences of the same opinion holder targeting at the same argument.\n"
                   "Text:\n"
                   "0. The United Nations Security Council on Friday renewed its peacekeeping mission in the disputed Western Sahara region for a year.\n"
                   "1. The money would boost Tokyo’s ability to oppose Beijing in the South China Sea and protect itself from a possible missile attack by North Korea.\n"
                   "2. Samantha Power acknowledged in remarks after the vote that the renewal of the Minurso mandate this year had been challenging and contentious.\n"
                   "3. Ms.Power told reporters that she was pleased that the resolution has passed.\n"
                   "4. But she was also concerned how soon the resolution might help diminish any lingering frictions between Mr. Ban’s office and the Moroccan government.\n"
                   "5. One senior council diplomat said should be an 'acceptable compromise'.\n"
                   "Opinions:"
                   "(1), (2, 4), (5)\n\n"
                   "Text: \n%s\n"
                   "Opinions:",

                   "Please extract opinion segments from the given text."
                   "Each opinion segment contains continuous one or more sentences of the same opinion holder targeting at the same argument.\n"
                   "Text:\n"
                   "0. The United Nations Security Council on Friday renewed its peacekeeping mission in the disputed Western Sahara region for a year.\n"
                   "1. The money would boost Tokyo’s ability to oppose Beijing in the South China Sea and protect itself from a possible missile attack by North Korea.\n"
                   "2. Samantha Power acknowledged in remarks after the vote that the renewal of the Minurso mandate this year had been challenging and contentious.\n"
                   "3. Ms.Power told reporters that she was pleased that the resolution has passed.\n"
                   "4. But she was also concerned how soon the resolution might help diminish any lingering frictions between Mr. Ban’s office and the Moroccan government.\n"
                   "5. One senior council diplomat said should be an 'acceptable compromise'.\n"
                   "Opinion tuples: (start index, end index)"
                   "(1, 1), (2, 4), (5, 5)\n\n"
                   "Text: \n%s\n"
                   "Opinion tuples: (start index, end index):",

                   "Please extract opinion segments from the given text."
                   "Each opinion segment contains continuous one or more sentences of the same opinion holder targeting at the same argument.\n"
                   "Text: \n%s\n"
                   "Please output opinion tuples. The format should be (start index, end index), such as (1, 1), (2, 4), (5, 5).",
                   ]
        prompt_idx = 1
        text_max_len = max_token_len - len(prompts[prompt_idx].split())
        sents = [str(idx) + '. ' + sent for idx, sent in enumerate(instance['document'])]
        saved_sents = []
        text_len = 0
        for sent in sents:
            text_len += len(sent.split())
            if text_len > text_max_len:
                break
            else:
                saved_sents.append(sent)
        text = '\n'.join(saved_sents)
        return prompts[prompt_idx] % text

    if dataset == 'APE':
        prompts = ["Please extract arguments from the given text. "
                   "Each argument contains continuous one or more sentences.\n\n"
                   "Text:\n"
                   "0. This paper studies the accuracy vs model-size trade-off of quantized CNNs under different channel width multipliers.\n"
                   "1. One of my main concerns is the direct usage of total number of bits as an equivalent measurement between models.\n"
                   "2. While it is useful to measure the storage cost for weights.\n"
                   "3. The choices of bit width will likely affect the computing cost in a non-trivial way, depending on the target hardware platform.\n"
                   "4. We thank the reviewer for their feedback and for finding our paper well-written and motivated.\n"
                   "5. Regarding your main concern, we agree that it is non-trivial to see if equal model size reflects "
                   "compute (in terms of latency and/or energy) since it depends on the application scenario and software/hardware implementation.\n"
                   "Arguments tuples: (start index, end index)\n"
                   "(1, 3), (5, 5)\n\n"
                   "Text: %s\n"
                   "Arguments tuples: (start index, end index)",

                   "Please extract arguments from the given text. "
                   "Each argument contains continuous one or more sentences.\n\n"
                   "Text: %s\n"
                   "Please output argument tuples. The format should be (start index, end index), such as (1, 1), (2, 4), (5, 5).",
                   ]
        prompt_idx = 0
        text_max_len = max_token_len - len(prompts[prompt_idx].split())
        sents = [str(idx) + '. ' + sent for idx, sent in enumerate(instance['document'])]
        saved_sents = []
        text_len = 0
        for sent in sents:
            text_len += len(sent.split())
            if text_len > text_max_len:
                break
            else:
                saved_sents.append(sent)
        text = '\n'.join(saved_sents)
        if len(saved_sents) != len(sents):
            if not slient:
                print(
                    'Truncated %d to %d tokens.' % (len('\n'.join(sents).split()), len('\n'.join(saved_sents).split())))
        return prompts[prompt_idx] % text

    if dataset == 'LitBank':
        prompts = ["Please answer the given question based on the context.\n\n"
                   "Context: %s\n"
                   "Question: %s. Identify all mentions of the same entity that (?) refers to in the following text.\n"
                   "Answer:",

                   "Please answer the given question based on the context.\n\n"
                   "Context: %s\n"
                   "Question: %s. Which expressions in the context refer to [MASK] ?\n"
                   "Answer:",

                   "Please answer the given question based on the context.\n\n"
                   "Context: There was a youthful private who listened with eager ears to the words of the tall soldier "
                   "and to the varied comments of his comrades. After receiving a fill of discussions concerning marches "
                   "and attacks , he went to his hut.\n"
                   "Question: There was [MASK] who listened with eager ears to the words of the tall soldier and to the varied comments of his comrades. "
                   "Which expressions in the context refer to the same entity with [MASK] ?\n"
                   "Answer: a youthful private, his, he, his\n\n"
                   "Context: %s\n"
                   "Question: %s. Which expressions in the context refer to the same entity with [MASK] ?\n"
                   "Answer:",

                   "Please answer the given question based on the context.\n\n"
                   "Context: He was a youthful private who listened with eager ears to the words of the tall soldier "
                   "and to the varied comments of his comrades. After receiving a fill of discussions concerning marches "
                   "and attacks , he went to his hut.\n"
                   "Question: [MASK] was a youthful private who listened with eager ears to the words of the tall soldier and to the varied comments of his comrades. "
                   "Which expressions in the context refer to [MASK] ?\n"
                   "Answer: He, a youthful private, his, he, his\n\n"
                   "Context: %s\n"
                   "Question: %s. Which expressions in the context refer to [MASK] ?\n"
                   "Answer:",
                   ]
        prompt_idx = 2
        question = str.replace(instance['question'], '( ? )', '[MASK]')
        context = ' '.join(instance['document']).split()
        context_max_len = max_token_len - len((prompts[prompt_idx] + question).split())
        if context_max_len < len(context):
            if not slient:
                print('Truncated %d to %d tokens.' % (len(context), context_max_len))
        context = ' '.join(context[:min(context_max_len, len(context))])
        return prompts[prompt_idx] % (context, question)

    if dataset == 'NarrativeQA':
        prompts = ["Please answer the given question based on the context."
                   "If the answer is not present in the context, please reply with 'unanswerable'.\n\n"
                   "Context: %s\n"
                   "Question: %s.\n"
                   "Answer:",
                   ]
        prompt_idx = 0
        question = instance['question']
        context = instance['document'].split()
        context_max_len = max_token_len - len((prompts[prompt_idx] + question).split())
        if context_max_len < len(context):
            if not slient:
                print('Truncated %d to %d tokens.' % (len(context), context_max_len))
        context = ' '.join(context[:min(context_max_len, len(context))])
        return prompts[prompt_idx] % (context, question)


def score(rs, dataset):
    if dataset == 'Hyperpartisan':
        filtered_rs = []

        # extract answer
        all_gold_labels = list(set([str(r['gold']) for r in rs]))
        for r_idx, r in enumerate(rs):
            rs[r_idx]['pred'] = str(rs[r_idx]['pred'])
            rs[r_idx]['gold'] = str(rs[r_idx]['gold'])
            for gold_label in all_gold_labels:
                if gold_label.lower() in r['pred'].lower():
                    rs[r_idx]['pred'] = gold_label

        for ridx, r in enumerate(rs):
            if r['pred'].lower() in ['true', 'false']:
                filtered_rs.append(r)
            else:
                print(r)
        total_num = len(filtered_rs)
        corr_num = sum([1 for r in rs if r['pred'].lower() == r['gold'].lower()])
        print('All examples: ', len(rs))
        print('Filtered examples: ', total_num)
        print('Acc: ', corr_num / total_num)
        return corr_num / total_num

    if dataset == 'ContractNLI':
        filtered_rs = []

        # extract answer
        all_gold_labels = list(set([r['gold'] for r in rs]))
        for r_idx, r in enumerate(rs):
            rs[r_idx]['pred'] = str(rs[r_idx]['pred'])
            rs[r_idx]['gold'] = str(rs[r_idx]['gold'])
            for gold_label in all_gold_labels:
                if gold_label.lower() in r['pred'].lower():
                    rs[r_idx]['pred'] = gold_label

        for r in rs:
            if r['pred'] in ['NotMentioned', 'Entailment', 'Contradiction']:
                filtered_rs.append(r)
            else:
                print(r)
        total_num = len(filtered_rs)
        corr_num = sum([1 for r in rs if r['pred'].lower() == r['gold'].lower()])
        print('All examples: ', len(rs))
        print('Filtered examples: ', total_num)
        print('Acc: ', corr_num / total_num)
        return corr_num / total_num

    if dataset == 'ECOM' or dataset == 'APE':
        # filter none
        filtered_rs = [r for r in rs if r['pred'] is not None]
        print(f'Filtered examples: {len(filtered_rs)} / {len(rs)}')

        is_segment_metric = False
        total_pred_num = 0
        total_gold_num = 0
        total_corr_num = 0
        for r in filtered_rs:
            pred_labels = re.findall(r'\((\d+), (\d+)\)', r['pred'])
            pred_labels = [(int(label[0]), int(label[1])) for label in pred_labels]
            if not is_segment_metric:
                new_pred_labels = []
                for label in pred_labels:
                    for idx in range(label[0], label[1] + 1):
                        new_pred_labels.append((idx, idx))
                pred_labels = new_pred_labels
            gold_labels = []
            for idx, label in enumerate(r['gold']):
                if is_segment_metric:
                    if label == "B":
                        start_idx = idx
                    if (label == "I" or label == "B") and (idx >= len(r['gold']) - 1 or r['gold'][idx + 1] != "I"):
                        end_idx = idx
                        gold_labels.append((start_idx, end_idx))
                else:
                    if label in ['B', 'I']:
                        gold_labels.append((idx, idx))
            pred_num = len(pred_labels)
            gold_num = len(gold_labels)
            corr_num = len(set(gold_labels).intersection(pred_labels))
            total_pred_num += pred_num
            total_gold_num += gold_num
            total_corr_num += corr_num
        p = total_corr_num / total_pred_num
        r = total_corr_num / total_gold_num
        f = 2 * p * r / (p + r)
        print(total_corr_num, total_pred_num, total_gold_num)
        print(p, r, f)
        return f

    if dataset == 'SummScreen' or dataset == 'GovReport':
        rouger = Rouge()
        pred_summs = [r['pred'] for r in rs]
        gold_summs = [r['gold'] for r in rs]

        # filter none
        filtered_idxs = [idx for idx, summ in enumerate(pred_summs) if summ is not None]
        print(f'Filtered examples: {len(filtered_idxs)} / {len(gold_summs)}')
        pred_summs = [summ for idx, summ in enumerate(pred_summs) if idx in filtered_idxs]
        gold_summs = [summ for idx, summ in enumerate(gold_summs) if idx in filtered_idxs]

        scores = rouger.get_scores(pred_summs, gold_summs, avg=True)
        print(scores)
        return scores['rouge-l']['f']

    if dataset == 'Qasper':
        f1s = []

        # filter none
        filtered_rs = [r for r in rs if r['pred'] is not None]
        print(f'Filtered examples: {len(filtered_rs)} / {len(rs)}')

        for r in filtered_rs:
            pred_answer, gold_answer = r['pred'], r['gold']
            # modified
            # boolean
            # if gold_answer.lower() == 'yes' or gold_answer.lower() == 'no':
            #     pred_answer = 'Yes' if 'yes' in pred_answer.lower() else pred_answer
            #     pred_answer = 'No' if 'no' in pred_answer.lower() else pred_answer
            f1 = token_f1_score(pred_answer, gold_answer)
            f1s.append(f1)
        mean = lambda x: sum(x) / len(x) if x else 0.0
        print(mean(f1s))
        return mean(f1s)

    if dataset == 'NarrativeQA':
        # none -> unanswerable
        for r_idx, r in enumerate(rs):
            for t in ['pred', 'gold']:
                if r[t] is None:
                    rs[r_idx][t] = 'unanswerable'
        f1s = []
        for r in rs:
            pred_answer, gold_answers = r['pred'], r['gold']
            gold_answers = ast.literal_eval(gold_answers)
            f1 = 0
            for gold_answer in gold_answers:
                this_f1 = token_f1_score(pred_answer, gold_answer)
                f1 = max(this_f1, f1)
            f1s.append(f1)
        mean = lambda x: sum(x) / len(x) if x else 0.0
        print(mean(f1s))
        return mean(f1s)

    if dataset == 'LitBank':

        # filter none
        filtered_rs = [r for r in rs if r['pred'] is not None]
        print(f'Filtered examples: {len(filtered_rs)} / {len(rs)}')

        # extract answer
        if 'The expressions that refer to the same entity with [MASK] are' in ' '.join([r['pred'] for r in filtered_rs]):  # vicuna-7b
            for idx, r in enumerate(filtered_rs):
                if '"' in r['pred']:
                    filtered_rs[idx]['pred'] = ','.join(re.findall(r'"(.*?)"', r['pred']))
        elif 'The expression' in ' '.join([r['pred'] for r in filtered_rs]):  # mistral
            for idx, r in enumerate(filtered_rs):
                if '"' in r['pred']:
                    filtered_rs[idx]['pred'] = ','.join(re.findall(r'"(.*?)"', r['pred']))

        pred_num = 0
        gold_num = 0
        corr_num = 0
        fs = []
        for r in filtered_rs:
            p_expressions = [e.strip().lower() for e in r['pred'].split(',')]
            g_expressions = [e.strip().lower() for e in r['gold'].split(',')]

            i_pred_num = len(p_expressions)
            i_gold_num = len(g_expressions)
            soft_match = lambda x, y: True if normalize_answer(x) in normalize_answer(y) or\
                                              normalize_answer(y) in normalize_answer(x) else False
            i_corr_num = 0
            for p in p_expressions:
                for g in g_expressions:
                    if soft_match(p, g):
                        i_corr_num += 1
                        break

            i_p = i_corr_num / i_pred_num
            i_r = i_corr_num / i_gold_num
            i_f = 2 * i_p * i_r / (i_p + i_r) if i_p + i_r else 0

            pred_num += i_pred_num
            gold_num += i_gold_num
            corr_num += i_corr_num
            fs.append(i_f)

        p = corr_num / pred_num
        r = corr_num / gold_num
        f_micro = 2 * p * r / (p + r)
        f_macro = sum(fs) / len(fs)
        print('Micro (avg instance): ', p, r, f_micro)
        print('Macro (avg category): ', sum(fs) / len(fs))
        return f_macro


def eval(dataset, sample_num, max_input_len, model, model_file, paras, only_test=False):
    print(f'{bcolors.SOFT_GREEN}Start test the performance of {model} on {dataset}!{bcolors.ENDC}')

    # read data
    input_path = f'input/{dataset}_{sample_num}.csv'
    if os.path.exists(input_path):
        data = read_file(input_path)
    else:
        examples = read_data(get_data_dir(dataset), dataset, sample_num)
        data = pd.DataFrame(examples)
        data['input'] = data.apply(lambda row: construct_input(row, dataset, max_input_len), axis=1)
        data = data[['id', 'input', 'label']]
        data.to_csv(input_path, index=False)
    print(f'Successfully load dataset {dataset}, len {len(data)}.')

    result_dir = f'result/{model}/{dataset}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = f'{result_dir}/result.csv'
    if os.path.exists(result_path):
        result_data = read_file(result_path)
    else:
        result_data = pd.DataFrame(columns=['id', 'gold', 'pred'])
        result_data['id'] = data['id'].tolist()
    result_data.set_index('id', inplace=True)

    # answer
    if not only_test:
        data.set_index('id', inplace=True)
        empty_result = result_data[result_data[f'pred'].isna()]
        empty_id = empty_result.index.tolist()
        if len(empty_id) > 0:
            print(f'{dataset}: Unanswered question: {len(empty_id)} / {len(result_data)}')
        unanswered_data = data.loc[empty_id]
        for answers in chat_with_llm_multi_thread(unanswered_data['input'].tolist(),
                                                  addition_information=empty_id,
                                                  model=model,
                                                  model_file=model_file,
                                                  paras=paras,
                                                  chunk_size=1,
                                                  thread_num=8,
                                                  max_new_tokens=max(data['label'].apply(lambda x: len(str(x).split())))):
            for answer in answers:
                idx = answer[1]
                a = answer[0]
                result_data.loc[idx, 'pred'] = a
                result_data.loc[idx, 'gold'] = data.loc[idx, 'label']
                # special answer format
                if dataset == 'LitBank':
                    result_data.loc[idx, 'gold'] = ', '.join([single_gold_answer[3] for single_gold_answer in ast.literal_eval(data.loc[idx, 'label'])])
                if (len(result_data) - sum(result_data['pred'].isna())) % 3 == 0:
                    result_data.reset_index(inplace=True)
                    write_file(result_path, result_data)
                    result_data.set_index('id', inplace=True)
        result_data.reset_index(inplace=True)
        write_file(result_path, result_data)
        result_data.set_index('id', inplace=True)

    # score
    json_data = result_data.to_json(orient='records')
    json_data = json.loads(json_data)
    return score(json_data, dataset)


def get_data_dir(dataset):
    if dataset in ['Hyperpartisan', 'ContractNLI']:
        data_dir = 'data/DocumentClassification'
    elif dataset in ['ECOM', 'APE']:
        data_dir = 'data/DocumentStructureAnalysis'
    elif dataset in ['LitBank', 'NarrativeQA']:
        data_dir = 'data/DocumentExtraction'
    elif dataset in ['SummScreen', 'GovReport', 'Qasper']:
        data_dir = 'data/Transcription'

    return data_dir


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def main():
    # constant
    datasets = ['Hyperpartisan', 'ContractNLI',
                'ECOM', 'APE',
                'LitBank', 'NarrativeQA',
                'SummScreen', 'GovReport', 'Qasper']

    # parameters
    sample_num = 50
    max_input_len = 5000
    models = ['gpt-4',
              'gpt-3.5-turbo',
              'vicuna-7b',
              'llama-2-7b-chat',
              'mistral-7b-instruct',
              'chatglm3-6b']
    paras = {'seed': 42, 'temperature': 0}
    only_test = False
    random.seed(42)

    # eval
    report = pd.DataFrame(columns=['model'] + datasets)
    report.set_index('model', inplace=True)

    for model in models:
        # load open model
        if model in open_models and not only_test:
            t1 = time.perf_counter()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_file = MyAutoModel(model, default_model_paths[model], None, device)
            t2 = time.perf_counter()
            print(f"{bcolors.SOFT_GREEN}Loading tokenizer and model: took {t2 - t1} seconds to execute.{bcolors.ENDC}")
        else:
            model_file = None

        # eval
        for dataset in datasets:
            dataset_score = eval(dataset, sample_num, max_input_len, model, model_file, paras)
            report.loc[model, dataset] = dataset_score

    report.to_csv('result/report.csv')


if __name__ == '__main__':
    main()
