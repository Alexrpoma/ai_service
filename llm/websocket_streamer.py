from transformers import AutoTokenizer, TextStreamer
import asyncio

class WebSocketStreamer(TextStreamer):
    """
    A TextStreamer that puts the generated tokens in an asyncio.Queue
    instead of printing to stdout.
    """
    def __init__(self, tokenizer: AutoTokenizer, queue: asyncio.Queue, skip_prompt: bool = True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)
        self.queue = queue
        self.stop_signal = None # To indicate the end of the stream

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put_nowait(text)
        if stream_end:
            self.queue.put_nowait(self.stop_signal) # End signal