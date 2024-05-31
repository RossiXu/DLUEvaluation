import os
os.environ['EVENTLET_NO_GREENDNS'] = 'yes'

import openai
from tqdm import tqdm
from utils.config import *
import eventlet
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import tiktoken

eventlet.monkey_patch(os=False)

openai.api_base = ""
openai.api_key = ""


def _chat_with_llm(model='gpt-3.5-turbo', message='', paras={}, max_new_tokens=100):
    # truncate
    if model == 'gpt-3.5-turbo':
        max_sequence_len = 16385
    else:
        max_sequence_len = 16385
    max_input_len = max_sequence_len - (max_new_tokens + 200)
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(message)
    if len(tokens) > max_input_len:
        print(f'Truncated from {len(tokens)} to {max_input_len}!')
        tokens = tokens[:max_input_len]
        message = encoding.decode(tokens)

    # answer
    if model in openai_chat_models:
        message = [{"role": "user", "content": message}] if type(message) is str else message
        completion = openai.ChatCompletion.create(
            model=model,
            messages=message,
            max_tokens=max_new_tokens,
            **paras
        )
        answer = completion['choices'][0]['message']['content']
        return answer
    elif model in openai_completion_models:
        completion = openai.Completion.create(
            model=model,
            prompt=message,
            **paras
        )
        answer = completion['choices'][0]['text']
        return answer


def chat_with_llm(model, message, addition_information=None, model_file=None,
                  retry_time=max_retry_time, run_time=max_run_time, paras={}, port=9222, max_new_tokens=100):
    """
    answer = [model_answer: str, additional_information: list]
    """
    # 如果失败或者超时，则重复尝试
    answer = [] if type(message) is list else ''
    for retry_idx in range(retry_time):
        message = message[len(answer):] if type(message) is list else message
        try:
            if model in openai_chat_models or model in openai_completion_models:
                if type(message) is list:
                    for single_message in message:
                        with eventlet.Timeout(run_time, True):
                            answer.append(_chat_with_llm(model, single_message, paras, max_new_tokens))
                else:
                    with eventlet.Timeout(run_time, True):
                        answer = _chat_with_llm(model, message, paras, max_new_tokens)
            elif model in open_models:
                answer = model_file.answer(message, max_new_tokens)
            break
        except eventlet.timeout.Timeout as te:
            # 处理 Timeout 异常
            print("Timeout exception:", te)
            print('Failed to chat with %s for the %s time.' % (model, str(retry_idx + 1)))
        except Exception as e:
            # 处理其他类型的异常
            print(type(e).__name__, e)
            print('Failed to chat with %s for the %s time.' % (model, str(retry_idx + 1)))
            # time.sleep(0.5)

    if addition_information:
        if type(answer) is str:
            answer = [answer, addition_information]
        else:
            answer = [[a, addition_information[a_idx]] for a_idx, a in enumerate(answer)]
    return answer


def chat_with_llm_multi_thread(messages, model='gpt-3.5-turbo', addition_information=None, model_file=None,
                               thread_num=8,
                               retry_time=max_retry_time, run_time=max_run_time,
                               output=None, paras={}, chunk_size=1, max_new_tokens=100):
    if not addition_information:
        addition_information = [None] * len(messages)
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        ports = [[9222 + i, True] for i in range(thread_num)]
        futures = [executor.submit(chat_with_llm, model, messages[start_idx:start_idx + chunk_size],
                                   addition_information[start_idx:start_idx + chunk_size], model_file, retry_time,
                                   run_time, output, paras, ports, max_new_tokens) for start_idx in range(0, len(messages), chunk_size)]
        for future in tqdm(concurrent.futures.as_completed(futures)):
            try:
                answer = future.result()  # 获取线程的结果
                yield answer
            except Exception as e:
                print(f"An error occurred: {e}")

