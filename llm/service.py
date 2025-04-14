import asyncio
import configparser
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .websocket_streamer import WebSocketStreamer


class Service:
    def __init__(self):
        config = configparser.ConfigParser()

        # --Uncomment if you need to login to Hugging Face--
        # from huggingface_hub import login
        # self.__auth_token = config.get("LLM", "AUTH_TOKEN")
        # login(self.__auth_token)

        # Load the model and tokenizer
        config.read('config.ini')
        self.__model_id = config.get("LLM", "MODEL")
        self.__prompt = config.get("LLM", "PROMPT")
        self.__tokenizers = AutoTokenizer.from_pretrained(self.__model_id)

        # Load the model with 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.__model = AutoModelForCausalLM.from_pretrained(
            self.__model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )

    def _blocking_generate(self, input_ids, streamer, max_length):
        logging.info("Starting blocking generation...")
        try:
            self.__model.generate(
                input_ids,
                streamer=streamer,
                max_new_tokens=max_length,
                eos_token_id=self.__tokenizers.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            logging.info("Blocking generation finished.")
        except Exception as e:
            logging.error(f"Error during model generation: {e}")
            streamer.queue.put_nowait(f"ERROR: {e}")
            streamer.queue.put_nowait(streamer.stop_signal)


    async def run_async_stream(self, qdrant_response: str, url: str, max_length: int, queue: asyncio.Queue):
        payload = f"summary: {qdrant_response} URL: {url}"
        messages = [
            {"role": "system", "content": self.__prompt},
            {"role": "user", "content": payload},
        ]
        try:
            input_ids = self.__tokenizers.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.__model.device)

            streamer = WebSocketStreamer(self.__tokenizers, queue, skip_prompt=True)

            # Execute the model generation in a separate thread
            logging.info("Dispatching generation to thread...")
            await asyncio.to_thread(
                self._blocking_generate,
                input_ids,
                streamer,
                max_length
            )
            logging.info("Thread execution supposedly finished.")
            # The end signal (None) is set by the streamer in on_finalized_text when stream_end=True

        except Exception as e:
            print(f"Error preparing for generation or in threading: {e}")
            await queue.put(f"ERROR: {e}")
            await queue.put(None)