import os
os.environ['HF_MODULES_CACHE'] = 'output'
os.environ['TRANSFORMERS_CACHE'] = 'output'
import torch
from transformers import AutoTokenizer, AutoModel
import transformers
from fastchat.model import load_model, get_conversation_template
import eventlet
eventlet.monkey_patch(os=False)

default_model_paths = {'vicuna-7b': '',
                       'vicuna-13b': '',
                       'alpaca-7b': '',
                       'llama-2-7b': '',
                       'llama-2-13b': '',
                       'llama-2-7b-chat': '',
                       'llama-2-13b-chat': '',
                       'mistral-7b': '',
                       'mistral-7b-instruct': '',
                       'chatglm3-6b': ''
                       }

default_model_series = {'vicuna': ['vicuna-7b', 'vicuna-13b'],
                        'llama': ['llama-2-7b', 'llama-2-13b', 'llama-2-7b-chat', 'llama-2-13b-chat'],
                        'mistral': ['mistral-7b', 'mistral-7b-instruct'],
                        'chatglm': ['chatglm3-6b']}

default_model_prompt_template = {'llama-2-7b-chat': '<s>[INST] {prompt} [/INST] {answer} </s>',
                                 'mistral-7b-instruct': '<s>[INST] {prompt} [/INST] {answer} </s>',
                                 'chatglm3-6b': '<|user|>\n{prompt}\n<|assistant|>\n{answer}'}


def get_model_series(model_name):
    for s, models in default_model_series.items():
        if model_name in models:
            return s

    return None


class Model:
    def __init__(self, model_name, model_path, ckpt_model_path, device):
        self.model_name = model_name
        self.model_path = model_path
        self.ckpt_model_path = ckpt_model_path
        self.device = device
        self.max_sequence_length = 4096

        self.__load__()

    def __load__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        try:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
        except:
            print('Cannot set pad_token.')

        if self.ckpt_model_path:
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path)
            print(f'Successfully load model {self.model_path}')
            state_dict = torch.load(self.ckpt_model_path, map_location='cpu')
            model.load_state_dict(state_dict['state'])
            print(f'Successfully load ckpt file {self.ckpt_model_path}')
            model = model.to(self.device)
            model.eval()
        else:
            model = self.model_path

        self.model = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def answer(self, message, max_new_tokens):
        messages = [message] if type(message) is str else message

        results = []
        for message in messages:
            # get input
            conv = get_conversation_template(self.model_path)
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # tokenize input
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

            # truncate
            max_input_len = self.max_sequence_length - (max_new_tokens + 100)
            input_ids = inputs["input_ids"][0]
            if len(input_ids) > max_input_len:
                print(f'Truncated from {len(input_ids)} to {max_input_len}')
                input_ids = input_ids[:max_input_len]
                inputs["input_ids"] = input_ids.unsqueeze(0)
                inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_len]
            inputs = inputs.to(self.device)

            # answer
            output_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens+100)

            # decode answer
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
            outputs = self.tokenizer.decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            result = outputs
            results.append(result)

        return results


class LlamaModel(Model):
    def __init__(self, model_name, model_path, ckpt_model_path, device):
        super(LlamaModel, self).__init__(model_name, model_path, ckpt_model_path, device)

    def __load__(self):
        model, self.tokenizer = load_model(self.model_path)

        if self.ckpt_model_path:
            state_dict = torch.load(self.ckpt_model_path, map_location='cpu')
            model.load_state_dict(state_dict['state'])
            print(f'Successfully load ckpt file {self.ckpt_model_path}')

        self.model = model.to(self.device)
        self.model.eval()


class VicunaModel(Model):
    def __init__(self, model_name, model_path, ckpt_model_path, device):
        super(VicunaModel, self).__init__(model_name, model_path, ckpt_model_path, device)

    def __load__(self):
        self.model, self.tokenizer = load_model(self.model_path)
        self.model = self.model.to(self.device)


class MistralModel(Model):
    def __init__(self, model_name, model_path, ckpt_model_path, device):
        super(MistralModel, self).__init__(model_name, model_path, ckpt_model_path, device)
        self.max_sequence_length = 32000
        self.max_sequence_length = 10000

    def __load__(self):
        model, self.tokenizer = load_model(self.model_path)

        if self.ckpt_model_path:
            state_dict = torch.load(self.ckpt_model_path, map_location='cpu')
            model.load_state_dict(state_dict['state'])
            print(f'Successfully load ckpt file {self.ckpt_model_path}')

        self.model = model.to(self.device)
        self.model.eval()


class ChatglmModel(Model):
    def __init__(self, model_name, model_path, ckpt_model_path, device):
        super(ChatglmModel, self).__init__(model_name, model_path, ckpt_model_path, device)
        self.max_sequence_length = 8000

    def __load__(self):
        model, self.tokenizer = load_model(self.model_path)

        if self.ckpt_model_path:
            state_dict = torch.load(self.ckpt_model_path, map_location='cpu')
            model.load_state_dict(state_dict['state'])
            print(f'Successfully load ckpt file {self.ckpt_model_path}')

        self.model = model.to(self.device)


class MyAutoModel(Model):
    def __init__(self, model_name, model_path, ckpt_model_path, device):
        super(MyAutoModel, self).__init__(model_name, model_path, ckpt_model_path, device)

    def __load__(self):
        model_series = get_model_series(self.model_name)
        if model_series == 'llama':
            self.model = LlamaModel(self.model_name, self.model_path, self.ckpt_model_path, self.device)
        elif model_series == 'vicuna':
            self.model = VicunaModel(self.model_name, self.model_path, self.ckpt_model_path, self.device)
        elif model_series == 'mistral':
            self.model = MistralModel(self.model_name, self.model_path, self.ckpt_model_path, self.device)
        elif model_series == 'chatglm':
            self.model = ChatglmModel(self.model_name, self.model_path, self.ckpt_model_path, self.device)
        else:
            raise ValueError(f'Inference of model {self.model_name} has not been implemented!')

    def answer(self, message, max_new_tokens):
        return self.model.answer(message, max_new_tokens)