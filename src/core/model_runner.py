import torch
import queue
import threading
from typing import Optional, Callable, List
from transformers import PreTrainedModel, PreTrainedTokenizer

class ModelRunner:
    """
    Handles asynchronous text generation from an LLM.
    Acts as the 'Producer' in the producer-consumer bridge.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        activation_queue: queue.Queue
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.activation_queue = activation_queue
        self._stop_event = threading.Event()

    def generate_async(self, prompt: str, max_new_tokens: int = 50):
        """Starts generation in a background thread."""
        thread = threading.Thread(
            target=self._generate,
            args=(prompt, max_new_tokens),
            daemon=True
        )
        thread.start()
        return thread

    def _generate(self, prompt: str, max_new_tokens: int):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Note: We assume hooks are already registered by the HookManager
        # in the AudioBridge which is watching the model.
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if self._stop_event.is_set():
                    break
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # In a real scenario, the HookManager (attached to the model)
                # captures the activation during the forward pass inside generate().
                # We don't manually push to the queue here; the HookManager's 
                # callback or a separate Bridge 'step' loop will handle it.
                
                # Update inputs for next token
                inputs['input_ids'] = outputs
                inputs['attention_mask'] = torch.ones_like(outputs)

    def stop(self):
        self._stop_event.set()
