import torch
torch.set_num_threads(4)
import zmq
import msgpack
from fastT5 import get_onnx_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import LLM_SERCER_PORT
import json
import time
import logging


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TextProcessorServer:
    def __init__(self):
        self.t5_mihm_v1_name ='./models/t5-base-finetuned-common_gen'
        self.t5_mihm_v1_name = './models/t5-mihm/'
        #self.cg_model_name = './t5-mihm-pt'
        self.t5_mihm_v1_model = get_onnx_model(self.t5_mihm_v1_name, self.t5_mihm_v1_name)
        #self.t5_mihm_v1_model = AutoModelForSeq2SeqLM.from_pretrained(self.cg_model_name)
        self.t5_mihm_v1_tok = AutoTokenizer.from_pretrained(self.t5_mihm_v1_name)
        
        self.t5_cg_name = './models/t5-base-finetuned-common_gen'
        self.t5_cg_model = get_onnx_model(self.t5_cg_name, self.t5_cg_name)
        self.t5_cg_tok = AutoTokenizer.from_pretrained(self.t5_cg_name)
        
        self.context = zmq.Context()
        self.total_time = 0
        self.run_count = 0
        
    def gen_t5(self, tokenizer: str, model:str, txt: str, config: dict):
        if not txt:
            return "NOTTTTT"
        start_time = time.perf_counter()
        t_input = txt
        token = getattr(self, tokenizer)(t_input, return_tensors='pt')
        print("OIYESINFERENCES")
        tokens = getattr(self, model).generate(
            input_ids=token['input_ids'],
            attention_mask=token['attention_mask'],
            num_beams=config.get('num_beams', 2),
            temperature=config.get('temperature', 1),
            do_sample=config.get('do_sample', True),
            max_new_tokens=config.get('max_new_tokens', 500),
            repetition_penalty=config.get('repetition_penalty', 1.05),
            no_repeat_ngram_size=config.get('no_repeat_ngram_size', 2),
        )
        output = getattr(self, tokenizer).decode(tokens.squeeze(), skip_special_tokens=True)
        end_time = time.perf_counter()
        # Update total time and count
        exec_time = end_time - start_time
        self.total_time += exec_time
        self.run_count += 1
        avg_time = self.total_time / self.run_count

        logger.info(f"Execution time: {exec_time:.6f} seconds, Average time: {avg_time:.6f} seconds over {self.run_count} runs")
        
        return output[2:-2]
    
    def gen_t5_cg(self, txt:str, config: dict):
        return self.gen_t5("t5_cg_tok", "t5_cg_model", txt, config)
        
    def gen_t5_mihm_cg(self, txt: str, config: dict):
        return self.gen_t5("t5_mihm_v1_tok", "t5_mihm_v1_model",  "grammar: " + txt, config)

    def gen(self, data: dict):
        logger.info(data)
        if data['type'] == 'ping':
            return ({'type': 'pong', 'from': 'llm server'})
        text = data['text']
        config = data.get('config', {})
        model = data.get('model')
        if model == "T5-mihm-gc":
            return self.gen_t5_mihm_cg(text, config)
        elif model == "T5-cg":
            return self.gen_t5_cg(text, config)
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
                    packed_response = msgpack.packb(response)
                    self.socket.send(packed_response)
                else:
                    packed_response = json.dumps(response)
                    self.socket.send_string(packed_response)
            except Exception as e:
                if "current state" in str(e):
                    self.socket.close()
                    self.socket = self.context.socket(zmq.REP)
                    self.socket.bind(f"tcp://127.0.0.1:{LLM_SERCER_PORT}")
                logger.error(e)
                logger.error(f"requesttt {data}")
                    
server = TextProcessorServer()
server.start_server()
