import torch
torch.set_num_threads(4)
import zmq
import msgpack
from fastT5 import get_onnx_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LongT5ForConditionalGeneration
from config import LLM_SERCER_PORT
from post_processors import POST_PORCESSORS
import json
import time
import logging
import os
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = 't5-mihm'
class TextProcessorServer:
    def __init__(self):
        self.context = zmq.Context()
        self.total_time = 0
        self.run_count = 0
        self.models = {}
        self.load_fastt5_models("./models/fastt5")
        self.load_torch_models("./models/torch")

    def load_fastt5_models(self, directory):
        """Dynamically load all models and tokenizers in the given directory."""
        for model_name in os.listdir(directory):
            model_path = os.path.join(directory, model_name)
            if os.path.isdir(model_path):
                try:
                    model = get_onnx_model(model_path, model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'generator': self.gen_t5
                    }
                    special_tokens = {
                        "additional_special_tokens": [
                            "</msg>", "</att_pic>", "</att_aud>", "</att_fil>", "</uid>", "</name>"
                        ]
                    }
                    # Add the special tokens to the tokenizer
                    tokenizer.add_special_tokens(special_tokens)
                    logger.info(f"loddddddddddddddddddddddd '{model_name}' from '{model_path}'")
                except Exception as e:
                    logger.error(f"NOTT! '{model_name}' at '{model_path}': {e}")
                    
    def load_torch_models(self, directory):
        """Dynamically load all models and tokenizers in the given directory."""
        for model_name in os.listdir(directory):
            model_path = os.path.join(directory, model_name)
            if os.path.isdir(model_path):
                try:
                    model = LongT5ForConditionalGeneration.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self.models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'generator': self.gen_t5
                    }
                    logger.info(f"loddddddddddddddddddddddd '{model_name}' from '{model_path}'")
                    print("Model class:", type(model).__name__)
                    print("Tokenizer class:", type(tokenizer).__name__)
                except Exception as e:
                    logger.error(f"NOTT! '{model_name}' at '{model_path}': {e}")

        
    def gen_t5(self, model, tokenizer, config: dict, txt: str,):
        if not txt:
            return "NOTTT"
        
        start_time = time.perf_counter()
        tokenized_input = tokenizer(
            txt, 
            return_tensors='pt', 
            padding=False,
            truncation=True,  # Enables truncation
            max_length=512    # Specify the maximum length
        )
        tokens = model.generate(
            input_ids=tokenized_input['input_ids'],
            attention_mask=tokenized_input['attention_mask'],
            num_beams=config.get('num_beams', 1),
            temperature=config.get('temperature', 1),
            do_sample=config.get('do_sample', True),
            max_new_tokens=config.get('max_new_tokens', 500),
            repetition_penalty=config.get('repetition_penalty', 1.05),
            no_repeat_ngram_size=config.get('no_repeat_ngram_size', 2),
        )
        output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        end_time = time.perf_counter()

        exec_time = end_time - start_time
        self.total_time += exec_time
        self.run_count += 1
        avg_time = self.total_time / self.run_count
        logger.info(f"Execution time: {exec_time:.6f}s, Average: {avg_time:.6f}s over {self.run_count} runs")

        return output

    
    def gen(self, data: dict):
        if data['type'] == 'ping':
            return ({'type': 'pong', 'from': 'llm server'})
        logger.info(data)
        text = data['text']
        config = data.get('config', {})
        model_name = data.get('model')
        if model_name in self.models:
            model = self.models[model_name]['model']
            tokenizer = self.models[model_name]['tokenizer']
            out = self.models[model_name]['generator'](model, tokenizer, config, text)
            if model_name in POST_PORCESSORS:
                return POST_PORCESSORS[model_name](out)
            logger.info(f"generation: {out}")
            return out
        else:
            model = self.models[DEFAULT_MODEL]['model']
            tokenizer = self.models[DEFAULT_MODEL]['tokenizer']
            out = self.models[DEFAULT_MODEL]['generator'](model, tokenizer, config, text)
            if DEFAULT_MODEL in POST_PORCESSORS:
                return POST_PORCESSORS[DEFAULT_MODEL](out)
            return out
        return 
        
    def start_server(self):
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://127.0.0.1:{LLM_SERCER_PORT}")
        print(f"SERCER PORT {LLM_SERCER_PORT}")
        while True:
            data = None
            try:
                message = self.socket.recv()
                data = None
                msg_type = None
                
                try:
                    data = msgpack.unpackb(message)
                    msg_type = 0
                except:
                    try:
                        data = json.loads(message.decode('utf-8'))
                        msg_type = 1
                    except:
                        self._logger.error("Failed to decode message as msgpack or JSON")
                        continue
                    
                response = self.gen(data)
                if msg_type == 0:
                    packed_response = msgpack.packb(response)#, flags=zmq.NOBLOCK)
                    self.socket.send(packed_response)
                else:
                    packed_response = json.dumps(response)
                    self.socket.send_string(packed_response)#, flags=zmq.NOBLOCK)
            except Exception as e:
                logger.error(traceback.format_exc())
                if "current state" in str(e):
                    self.socket.close()
                    self.socket = self.context.socket(zmq.REP)
                    self.socket.bind(f"tcp://127.0.0.1:{LLM_SERCER_PORT}")
                logger.error(e)
                logger.error(f"requesttt {data}")
                    
server = TextProcessorServer()
server.start_server()
