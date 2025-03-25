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
from llama_cpp import Llama

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = 't5-mihm'
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "</msg>", "</att_pic>", "</att_aud>", "</att_fil>", "</uid>", "</name>", "</end>"
    ]
}
class TextProcessorServer:
    def __init__(self):
        self.context = zmq.Context()
        self.total_time = 0
        self.run_count = 0
        self.models = {}
        self.load_fastt5_models("./models/fastt5")
        self.load_torch_models("./models/torch")
        self.load_llama_models("./models/llama")

    def load_llama_models(self, directory):
        """Dynamically load all LlamaCpp models in the given directory."""
        for model_name in os.listdir(directory):
            model_path = os.path.join(directory, model_name)
            if model_path.endswith('.gguf'):
                try:
                    model = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_threads=4
                    )
                    self.models[model_name.replace('.gguf', '')] = {
                        'model': model,
                        'tokenizer': None,  # LlamaCpp handles tokenization internally
                        'generator': self.gen_llama
                    }
                    logger.info(f"Loaded LlamaCpp model '{model_name}' from '{model_path}'")
                except Exception as e:
                    logger.error(f"Failed to load LlamaCpp model '{model_name}' at '{model_path}': {e}")

    def gen_llama(self, model: Llama, tokenizer: None, config, txt: str):
        if not txt:
            return None
        start_time = time.perf_counter()
        generation_config = {
            "max_tokens": config.get('max_new_tokens', 500),
            "temperature": config.get('temperature', 1.0),
            "top_p": config.get('top_p', 1.0),
            "top_k": config.get('top_k', 50),
            "repeat_penalty": config.get('repetition_penalty', 1.0),
            "stop": config.get('stop_sequences', []), 
            "echo": False, 
        }
        
        try:
            output = model(
                prompt=txt,
                **generation_config
            )
            generated_text = output['choices'][0]['text']
            end_time = time.perf_counter()
            exec_time = end_time - start_time
            self.total_time += exec_time
            self.run_count += 1
            avg_time = self.total_time / self.run_count
            logger.info(f"LlamaCpp execution time: {exec_time:.6f}s, Average: {avg_time:.6f}s over {self.run_count} runs")
            return generated_text
        except Exception as e:
            logger.error(f"Error during LlamaCpp generation: {e}")
            return None
        
    def load_fastt5_models(self, directory):
        """Dynamically load all models and tokenizers in the given directory."""
        for model_name in os.listdir(directory):
            model_path = os.path.join(directory, model_name)
            if os.path.isdir(model_path):
                try:
                    model = get_onnx_model(model_path, model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)#, use_fast = True, from_slow = True, legacy = False)
                    tokenizer_new = AutoTokenizer.from_pretrained(model_path, use_fast = True, from_slow = True, legacy = False)
                    self.models[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'tokenizer_new': tokenizer_new,
                        'generator': self.gen_t5
                    }
                    print(tokenizer.tokenize("<buferia>"))  # Should output ['<buferia>'], not split into subwords
                    # Check token ID
                    token_id = tokenizer.convert_tokens_to_ids("<buferia>")
                    print(f"Token ID of <buferia>: {token_id}")  # Should NOT return UNK (e.g., 100)
                    
                    #tokenizer.add_special_tokens(SPECIAL_TOKENS)
                    logger.info(f"loddddddddddddddddddddddd '{model_name}' from '{model_path}'")
                except Exception as e:
                    logger.error(f"NOTT! '{model_name}' at '{model_path}': {e}")
                    
    def load_torch_models(self, directory):
        """Dynamically load all models and tokenizers in the given directory."""
        for model_name in os.listdir(directory):
            model_path = os.path.join(directory, model_name)
            if os.path.isdir(model_path):
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True, from_slow = True, legacy = False)
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

    def truncate_from_start(self, text_, tokenizer, max_length):
        # Tokenize the input text without truncation
        task, text = text_.split(":", 1)
        tokens_t = tokenizer.tokenize(text)
        task_t = tokenizer.tokenize(task)
        # If the number of tokens exceeds max_length, truncate from the start
        if len(tokens_t) > max_length-len(task_t)-1:
            max_l = max_length-len(task_t)-1-4
            tokens_t = tokens_t[-max_l:]

        # Convert tokens back to a string
        truncated_text = tokenizer.convert_tokens_to_string(tokens_t)
        truncated_text = f"{task}:{truncated_text}"
        tokens_t = tokenizer.tokenize(truncated_text)
        if len(tokens_t) > max_length:
            print(len(tokens_t))
            print(truncated_text)
            raise ValueError("XD! ANGERS!")
        return truncated_text
    
    def gen_t5(self, model, tokenizer, config: dict, txt: str,):
        if not txt:
            return None#"NOTTT"
        if config.get("truncate_from_start", False):
            tmp = self.truncate_from_start(txt, tokenizer, max_length=512)
            txt = tmp
        start_time = time.perf_counter()
        tokenized_input = []
        if config.get("new_tokenizer", False):
            tokenized_input = tokenizer(
                txt, 
                return_tensors='pt', 
                padding=False,
                truncation=True,
                max_length=512
            )   
        else:
            tokenized_input = tokenizer(
                txt, 
                return_tensors='pt', 
                padding=False,
                truncation=True,
                max_length=512 
            )
        truncated_text = tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=config.get('skip_special_tokens_in', False))
        print("ACTUL INPUT=================================")
        print(truncated_text)
        print("ACTUL INPUT=================================")
        tokens = model.generate(
            input_ids=tokenized_input['input_ids'],  # The tokenized input text to generate output from.
            attention_mask=tokenized_input['attention_mask'],  # Attention mask to avoid attending to padding tokens.
            
            # Number of beams for beam search. More beams give better quality but are slower.
            num_beams=config.get('num_beams', 1),  # Defaults to 1 (no beam search, i.e., greedy generation).
            # Temperature controls randomness. Higher temperature increases randomness (1.0 is neutral).
            temperature=config.get('temperature', 1),  # Defaults to 1.0 (no change in probabilities).
            # Whether or not to use sampling (True means sampling, False means greedy decoding).
            do_sample=config.get('do_sample', True),  # Defaults to True (sampling is used).
            # Top-p (nucleus sampling) filters tokens to the smallest set with cumulative probability >= top_p.
            top_p=config.get('top_p', 1.0),  # Defaults to 1.0 (no filtering based on cumulative probability).
            # Top-k sampling limits to the top-k most likely tokens.
            top_k=config.get('top_k', 50),  # Defaults to 50 (limits to top 50 tokens).
            # Minimum probability for tokens to be sampled, scaled by the highest token probability.
            min_p=config.get('min_p', None),  # Defaults to None (no minimum threshold).
            # Typicality filter controls how typical a token is based on the conditional probability.
            typical_p=config.get('typical_p', 1.0),  # Defaults to 1.0 (no filtering based on typicality).
            # The maximum number of tokens to generate.
            max_new_tokens=config.get('max_new_tokens', 500),  # Defaults to 500 (max 500 new tokens).
            # A penalty applied to repetitive token generation.
            repetition_penalty=config.get('repetition_penalty', 1.0),  # Defaults to 1.0 (mild penalty for repetition).
            # Size of the n-grams that should not be repeated in the generated text.
            no_repeat_ngram_size=config.get('no_repeat_ngram_size', 2),  # Defaults to 2 (prevents repeating bigrams).
            # Diversity penalty for beam search to reduce repetitive behavior across beams.
            diversity_penalty=config.get('diversity_penalty', 0.0),  # Defaults to 0.0, no penalty on diversity.
            # Encoder repetition penalty (to discourage repeating original input in decoder generation).
            encoder_repetition_penalty=config.get('encoder_repetition_penalty', 1.0),  # Defaults to 1.0.
            # A length penalty that encourages shorter (negative) or longer (positive) sequences.
            length_penalty=config.get('length_penalty', 1.0),  # Defaults to 1.0 (neutral length penalty).
            # Set ngram size for repeated ngrams that should not appear.
            bad_words_ids=config.get('bad_words_ids', None),  # Defaults to empty list (no bad words filtered).
            # A list of token IDs that must be included in the generated text.
            force_words_ids=config.get('force_words_ids', None),  # Defaults to empty list (no forced words).
            # Option to renormalize logits after applying all the logit processors.
            renormalize_logits=config.get('renormalize_logits', False),  # Defaults to False (no renormalization).
            # Custom constraints that can be applied to token generation.
            constraints=config.get('constraints', None),  # Defaults to empty list (no constraints).
            # Force a specific token ID to appear at the beginning of the generated sequence.
            #forced_bos_token_id=config.get('forced_bos_token_id', model.config.forced_bos_token_id),  # Defaults to model config.
            # Force a specific token ID to appear at the end of the generated sequence.
            #forced_eos_token_id=config.get('forced_eos_token_id', model.config.forced_eos_token_id),  # Defaults to model config.
            # Option to remove invalid values (such as NaN or Inf) from the logits to prevent crashes.
            remove_invalid_values=config.get('remove_invalid_values', False),  # Defaults to False (no removal).
            # Exponential decay of length penalty for sequences after a certain token count.
            exponential_decay_length_penalty=config.get('exponential_decay_length_penalty', None),  # Defaults to None (no decay).
            # List of tokens to suppress in the generated output.
            suppress_tokens=config.get('suppress_tokens', None),  # Defaults to empty list (no suppressed tokens).
            # List of tokens to suppress only at the beginning of the generated output.
            begin_suppress_tokens=config.get('begin_suppress_tokens', None),  # Defaults to empty list (no initial suppression).
            # List of forced decoder token IDs that must appear at specific positions in the sequence.
            forced_decoder_ids=config.get('forced_decoder_ids', None),  # Defaults to empty list (no forced decoder IDs).
            # Biasing specific sequences of tokens to be more likely or unlikely to appear.
            sequence_bias=config.get('sequence_bias', None),  # Defaults to empty dict (no sequence bias).
            # Option to use token healing to improve token completion when there is greedy tokenization.
            token_healing=config.get('token_healing', False),  # Defaults to False (no token healing).
            # Guidance scale for classifier-free guidance (CFG). A higher scale forces the model to stick more closely to the input prompt.
            guidance_scale=config.get('guidance_scale', 1.0),  # Defaults to 1.0 (no guidance scale).
            # Switch to reduce peak memory usage with sequential beam search and sequential top-k for contrastive search.
            low_memory=config.get('low_memory', False),  # Defaults to False (no memory-saving mode).
            # Watermarking configuration to embed a small bias in generated tokens for traceability.
            watermarking_config=config.get('watermarking_config', None)  # Defaults to None (no watermarking).
        )
        
        output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=config.get('skip_special_tokens_out', False))
        end_time = time.perf_counter()

        exec_time = end_time - start_time
        self.total_time += exec_time
        self.run_count += 1
        avg_time = self.total_time / self.run_count
        logger.info(f"Execution time: {exec_time:.6f}s, Average: {avg_time:.6f}s over {self.run_count} runs")

        return output

    def processka_command(self, data: dict):
        if data['type'] == 'ping':
            return True, ({'type': 'pong', 'from': 'llm server'})
        if data['type'] == 'get_models':
            return True, list(self.models.keys())
        return False, None
    
    def gen(self, data: dict):
        is_cmd, out = self.processka_command(data)
        if is_cmd:
            return out
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
