# 通用设置
max_run_time = 10
max_retry_time = 1

# 常量
openai_chat_models = ['gpt-3.5-turbo',
                      'gpt-3.5-turbo-0613',
                      'gpt-3.5-turbo-16k',
                      'gpt-3.5-turbo-16k-0613',
                      'gpt3.5-turbo-0301',
                      'gpt-4-0613']
openai_chat_model_args = ['frequency_penalty', 'logit_bias', 'logprobs', 'top_logprobs',
                          'max_tokens', 'n', 'presence_penalty', 'response_format', 'seed', 'stop', 'stream',
                          'temperature', 'top_p', 'tools', 'tool_choice', 'user', 'function_call', 'functions']
openai_completion_models = ['text-davinci-003',
                            'text-davinci-002',
                            'text-davinci-001',
                            'text-curie-001',
                            'text-babbage-001',
                            'text-ada-001',
                            'davinci',
                            'curie',
                            'babbage',
                            'ada']
open_models = ['vicuna-7b',
               'vicuna-13b',
               'llama-2-7b-chat',
               'llama-2-13b-chat',
               'mistral-7b-instruct',
               'chatglm3-6b']