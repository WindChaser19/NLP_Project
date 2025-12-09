import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import datetime
import argparse
import json
import math
import os
import evaluate
import sys

#Qwen3 Stuff
from transformers.models.qwen3 import modeling_qwen3, modular_qwen3

from collections.abc import Callable
from typing import Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

###################################################################################################
# Dual position patching
###################################################################################################
def Qwen3Model_forward_patch(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_2: Optional[torch.LongTensor] = None, #############ADD PARAMETER
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        #########PATCH THIS
        if position_ids_2 is None:
            #print("Patched Qwen3 Forward: Only one position ID found")
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            #print("Patched Qwen3 Forward: Using 2 Position Ids")
            #gen_pos_2 = position_ids_2[:, -1:].item() + 1
            gen_pos_2 = 0
            position_embeddings = self.rotary_emb(hidden_states, position_ids, position_ids_2, gen_pos_2)
        ##########################################################REST SHOULDNT REALLY MATTER


        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

def Qwen3Causal_forward_patch(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_2:Optional[torch.LongTensor] = None, #######################################ADD THIS
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_ids_2 = position_ids_2, ###################################FEED INTO MODEL
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


###################################################################################################
# ROPE FORWARD PLUGINS
###################################################################################################

def hirope_forward_dual(self, x, position_ids, position_ids_2=None, gen_pos_2=None):
    d = x.shape[-1]
    half = d // 2

    # Split inv_freq for first and second half
    inv_freq1 = self.inv_freq[:half]
    inv_freq2 = self.inv_freq[half:]

    batch_size, seq_len = position_ids.shape
    
    if position_ids_2 is None:
        position_ids_2 = position_ids // 128

    if position_ids_2.shape != position_ids.shape:
        #print("SUMROPE FORWARD 2: overriding position ids 2 to ", gen_pos_2)
        if gen_pos_2 is not None:
            position_ids_2 = torch.full_like(position_ids, gen_pos_2)
        else:
            position_ids_2 = torch.zeros_like(position_ids)

    pos = position_ids[:, None, :].float()  # [B, 1, L]
    pos2 = position_ids_2[:, None, :].float()
    pos_2 = pos2 // 128 
    
    ####to conduct abalation, set pos2 as 0 
    ##pos2 = torch.zeros_like(pos) for ex. 
    
    if len(pos) > 1:
        print("pos: ", pos[:10])
        print("pos 2 : ", pos2[:10])
    # pos_2 = pos_2[:, None, :].float()

    freqs1 = (inv_freq1[None, :, None].float() @ pos).transpose(1, 2)  # [B, L, D/2]
    freqs2 = (inv_freq2[None, :, None].float() @ pos2).transpose(1, 2)

    # Duplicate like original RoPE (cos/sin both halves)
    freqs = torch.cat([freqs1, freqs2], dim = -1)
    emb = torch.cat((freqs, freqs), dim  = -1)

    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def hirope_forward(self, x, position_ids):
    d = x.shape[-1]
    half = d // 2

    # Split inv_freq for first and second half
    inv_freq1 = self.inv_freq[:half]
    inv_freq2 = self.inv_freq[half:]

    batch_size, seq_len = position_ids.shape

    pos = position_ids[:, None, :].float()  # [B, 1, L]
    pos_2 = position_ids // 128
    if len(pos) > 1:
        print("pos: ", pos[:10])
        print("pos 2 : ", pos_2[:10])
    pos_2 = pos_2[:, None, :].float()

    freqs1 = (inv_freq1[None, :, None].float() @ pos).transpose(1, 2)  # [B, L, D/2]
    freqs2 = (inv_freq2[None, :, None].float() @ pos_2).transpose(1, 2)

    # Duplicate like original RoPE (cos/sin both halves)
    freqs = torch.cat([freqs1, freqs2], dim = -1)
    emb = torch.cat((freqs, freqs), dim  = -1)


    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def sumrope_forward_dual(self, x, position_ids, position_ids_2=None, gen_pos_2=None):
    d = x.shape[-1]
    half = d // 2
    batch_size, seq_len = position_ids.shape
    
    if position_ids_2 is None:
        position_ids_2 = position_ids // 128

    if position_ids_2.shape != position_ids.shape:
        #print("SUMROPE FORWARD 2: overriding position ids 2 to ", gen_pos_2)
        if gen_pos_2 is not None:
            position_ids_2 = torch.full_like(position_ids, gen_pos_2)
        else:
            position_ids_2 = torch.zeros_like(position_ids)

    pos = position_ids[:, None, :].float()  # [B, 1, L]
    pos_2 = position_ids_2[:, None, :].float()

    freqs1 = (self.inv_freq[None, :, None].float() @ pos).transpose(1, 2)  # [B, L, D/2]
    freqs2 = (self.inv_freq[None, :, None].float() @ pos_2).transpose(1, 2)

    freqs = (freqs1 + freqs2)
    emb = torch.cat((freqs, freqs), dim  = -1)

    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling 
    #cos = emb.cos()
    #sin = emb.sin() 

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def sumrope_forward(self, x, position_ids):
    d = x.shape[-1]
    half = d // 2
    batch_size, seq_len = position_ids.shape

    pos = position_ids[:, None, :].float()  # [B, 1, L]
    pos_2 = position_ids // 128
    pos_2 = pos_2[:, None, :].float()

    freqs1 = (self.inv_freq[None, :, None].float() @ pos).transpose(1, 2)  # [B, L, D/2]
    freqs2 = (self.inv_freq[None, :, None].float() @ pos_2).transpose(1, 2)

    freqs = (freqs1 + freqs2)
    emb = torch.cat((freqs, freqs), dim  = -1)

    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling 
    #cos = emb.cos()
    #sin = emb.sin() 

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def sumhirope_forward(self, x, position_ids):
    d = x.shape[-1]
    half = d // 2

    # Split inv_freq for first and second half
    inv_freq1 = self.inv_freq[:half]
    inv_freq2 = self.inv_freq[half:]

    batch_size, seq_len = position_ids.shape

    pos = position_ids[:, None, :].float()  # [B, 1, L]
    pos_2 = position_ids // 128
    if len(pos) > 1:
        print("pos: ", pos[:10])
        print("pos 2 : ", pos_2[:10])
    pos_2 = pos_2[:, None, :].float()

    freqs1 = (inv_freq1[None, :, None].float() @ pos).transpose(1, 2)  # [B, L, D/2]
    freqs12 = (inv_freq2[None, :, None].float() @ pos).transpose(1, 2)  # [B, L, D/2]
    freqs2 = (inv_freq2[None, :, None].float() @ pos_2).transpose(1, 2)

    # Duplicate like original RoPE (cos/sin both halves)
    freqs = torch.cat([freqs1, freqs12 + freqs2], dim = -1)
    emb = torch.cat((freqs, freqs), dim  = -1)


    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling


    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def sumhirope_forward_dual(self, x, position_ids, position_ids_2=None, gen_pos_2=None):
    d = x.shape[-1]
    half = d // 2

    # Split inv_freq for first and second half
    inv_freq1 = self.inv_freq[:half]
    inv_freq2 = self.inv_freq[half:]

    batch_size, seq_len = position_ids.shape
    
    if position_ids_2 is None:
        position_ids_2 = position_ids // 128

    if position_ids_2.shape != position_ids.shape:
        #print("SUMROPE FORWARD 2: overriding position ids 2 to ", gen_pos_2)
        if gen_pos_2 is not None:
            position_ids_2 = torch.full_like(position_ids, gen_pos_2)
        else:
            position_ids_2 = torch.zeros_like(position_ids)


    if position_ids_2.shape != position_ids.shape:
        #print("SUMROPE FORWARD 2: overriding position ids 2 to ", gen_pos_2)
        if gen_pos_2 is not None:
            position_ids_2 = torch.full_like(position_ids, gen_pos_2)
        else:
            position_ids_2 = torch.zeros_like(position_ids)

    pos = position_ids[:, None, :].float()  # [B, 1, L]
    pos_2 = position_ids_2[:, None, :].float()

    if len(pos) > 1:
        print("pos: ", pos[:10])
        print("pos 2 : ", pos_2[:10])

    freqs1 = (inv_freq1[None, :, None].float() @ pos).transpose(1, 2)  # [B, L, D/2]
    freqs12 = (inv_freq2[None, :, None].float() @ pos).transpose(1, 2)  # [B, L, D/2]
    freqs2 = (inv_freq2[None, :, None].float() @ pos_2).transpose(1, 2)

    # Duplicate like original RoPE (cos/sin both halves)
    freqs = torch.cat([freqs1, freqs12 + freqs2], dim = -1)
    emb = torch.cat((freqs, freqs), dim  = -1)


    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling


    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


###################################################################################################
# MODEL / EVAL UTILS
###################################################################################################

def load_model(model_name, quant=None):
    print(f"[INFO] Loading model: {model_name}")
    kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if quant == "4bit":
        print("[INFO] Loading model in 4-bit mode...")
        kwargs.update({
            "load_in_4bit": True,
        })
    elif quant == "8bit":
        print("[INFO] Loading model in 8-bit mode...")
        kwargs.update({
            "load_in_8bit": True,
        })

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


###################################################################################################
# Copied from https://github.com/huggingface/datasets/blob/d3c7b9481d427ce41256edaf6773c47570f06f3b/metrics/squad/evaluate.py
# ^ above is ZeroSCROLLS code

import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

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

def truncate_repetition_ngram(pred, n=8):
    tokens = pred.split()         # split into words
    seen_ngrams = set()
    result = []
    for i in range(len(tokens)):
        if i + n > len(tokens):
            result.append(tokens[i])
            continue
        ngram = tuple(tokens[i:i+n])   # current sequence of n words
        if ngram in seen_ngrams:
            break                     # stop if repeated
        seen_ngrams.add(ngram)
        result.append(tokens[i])
    return ' '.join(result)

def remove_answer_prefix(text):
    """
    Remove a leading 'The answer is' (case-insensitive) from the text.
    Only removes it if it is at the very beginning.
    """
    if not text:
        return ""
    # Remove one occurrence at the start
    cleaned = re.sub(r'^\s*The answer is\s*', '', text, flags=re.IGNORECASE)
    return cleaned.strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0t
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
        
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_f1(predictions, references):
    f1 = 0
    for prediction, ground_truths in zip(predictions, references):
        pred = truncate_repetition_ngram(prediction)
        pred = remove_answer_prefix(pred)
        f1 += metric_max_over_ground_truths(f1_score, pred, ground_truths)
        # print("Pred:", pred)
        # print("Refs: ", ground_truths)
        # print("F1:", metric_max_over_ground_truths(f1_score, pred, ground_truths))
    return 100.0 * f1 / len(predictions)

def compute_f1_and_save_results(predictions, references, args): 
    modelsz = ""
    
    if args.model == "Qwen/Qwen3-1.7B":
        modelsz="17B"
    elif args.model == "Qwen/Qwen3-0.6B":
        modelsz="06B"
    elif args.model == "Qwen/Qwen3-4B":
        modelsz="4B"
    
    
    qualitative_file_name = f"newds_narrative_qa_generation_results_{args.rope}_{modelsz}.txt"
    f1 = 0
    
    
    with open(qualitative_file_name, 'a') as f:
        f.write("\n=== EVAL CONFIGS ===\n")
        f.write(f"Samples: {args.n}\n")
        f.write(f"Quantization: {args.quant}\n")
        f.write(f"Long summarization: {args.long}\n")
        
        num_qual_printed = 0
        
    
        for prediction, ground_truths in zip(predictions, references):
            pred = truncate_repetition_ngram(prediction)
            pred = remove_answer_prefix(pred)
            f1toadd  =  metric_max_over_ground_truths(f1_score, pred, ground_truths)
            f1 += f1toadd
            
            if num_qual_printed <= 20:
                f.write("\n################\n")
                f.write(f"Prediction:\n {prediction}\n")
                f.write(f"Truncated Prediction:\n {pred}\n")
                f.write(f"Ground Truth:\n {ground_truths}\n")
                f.write(f"F1 Score: {f1toadd}\n")
                num_qual_printed += 1
            
    final_f1 =  100.0 * f1 / len(predictions)
    save_results_to_file({"f1": final_f1}, args, filename = f"newds_narrative_qa_results_{args.rope}.txt")
    return final_f1


def save_results_to_file(scores, args, filename=None):
    """
    Save evaluation results to a text file

    Args:
        scores: Dictionary containing ROUGE scores
        args: Command line arguments
        filename: Optional filename
    """
    if filename is None:
        filename = f"eval_results_hirope.txt"

    with open(filename, 'a') as f:
        f.write("\n=== EVALUATION RESULTS ===\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Samples: {args.n}\n")
        f.write(f"Quantization: {args.quant}\n")
        f.write(f"Long summarization: {args.long}\n")
        f.write("\n--- ROUGE Scores ---\n")

        for k, v in scores.items():
            f.write(f"{k}: {v:.4f}\n")

        # Add a JSON section for easy parsing
        f.write("\n--- JSON Data ---\n")
        json.dump({
            "model": args.model,
            "samples": args.n,
            "quantization": args.quant,
            "long_summarization": args.long,
            "scores": scores,
        }, f, indent=2)

    print(f"[INFO] Results saved to: {filename}")
    return filename


##########################################################################


def test_narrativeqa(model, tokenizer, args):
    print("[INFO] Loading NarrativeQA dataset...")
    dataset = load_dataset(
    "json",
        data_files={
            "validation": os.path.join(".", "narrative_qa/validation.jsonl"),
            "test": os.path.join(".", "narrative_qa/test.jsonl"),
        }
    )

    preds = []
    refs = []

    print(f"[INFO] Running evaluation on {args.n} samples...")
    
    

    for i in tqdm(range(args.n)):
        sample = dataset["test"][i]
        sampleinput = sample["input"]
        all_answers = sample["output"]  
        
        max_tokens = 8192
        
        tokenized_full = tokenizer(sampleinput, return_tensors="pt")
        tokenized_input_full = tokenized_full.input_ids.to(model.device)
        attention_mask = tokenized_full.attention_mask.to(model.device)
        
        if tokenized_input_full.shape[1] > max_tokens:
            separator_and_query_text = sample['truncation_seperator'] + sample["input"][sample['query_start_index']:]
            tokenized_sep_query = tokenizer(separator_and_query_text, return_tensors="pt").to(model.device)
            input_without_query = sample['input'][:sample['query_start_index']]
            tokenized_input_without_query = tokenizer(input_without_query, return_tensors="pt").to(model.device)

            tokenized_input_without_query.input_ids = tokenized_input_without_query.input_ids[:, :max_tokens - tokenized_sep_query.input_ids.shape[1]]
            tokenized_input_without_query.attention_mask = tokenized_input_without_query.attention_mask[:, :max_tokens - tokenized_sep_query.attention_mask.shape[1]]

            tokenized_input = torch.cat([tokenized_input_without_query.input_ids, tokenized_sep_query.input_ids], dim=1)
            attention_mask = torch.cat([tokenized_input_without_query.attention_mask, tokenized_sep_query.attention_mask], dim=1)
            
    
        inputs = tokenized_input

        with torch.no_grad():
            output = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=100,        # sufficient for most answers
                do_sample=False,            # deterministic
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        decoded_input = tokenizer.decode(inputs[0], skip_special_tokens=True)
        pred = decoded[len(decoded_input):].strip()

        preds.append(pred)
        refs.append(all_answers)

    print("[INFO] Computing Scrolls/SQuAD-style F1...")
    final_f1 = compute_f1_and_save_results(preds, refs, args)
    print("\n=== FINAL RESULTS ===")
    print(f"F1: {final_f1:.4f}")

##########################################################################################
def summarize(model, tokenizer, doc, max_input=4096, max_new=512,dual="dual"):
    
    prompt_start = "Summarize the following government report:\n\n"
    
    prompt_start_inputs = tokenizer(
            prompt_start,
            return_tensors="pt",
            truncation=True,
            # max_length=max_input
        ).to(model.device)
        
    batch_size, seq_len = prompt_start_inputs['input_ids'].shape
    
    tokenized_list = [prompt_start_inputs]
    position_ids2_list = [torch.zeros(seq_len, dtype=torch.long).to(model.device)]
    cur_pos2 = 0
    
    for i in range(len(doc["title"])):
        title_name = doc["title"][i]
        paragraph = doc["paragraphs"][i]
        title_depth = doc["depth"][i]
        
        if title_depth == 1:
            cur_pos2 += 1
        
        paragraph_inputs = tokenizer(
            paragraph,
            return_tensors="pt",
            truncation=True,
        ).to(model.device)
        
        batch_size, seq_len = paragraph_inputs['input_ids'].shape
        tokenized_list.append(paragraph_inputs)
        position_ids2_list.append(torch.ones(seq_len, dtype=torch.long).to(model.device) * cur_pos2)
    
    
    cur_pos2 += 1
    prompt_end = "\n\nSummary:"
    prompt_end_inputs = tokenizer(
        prompt_end,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)
        
    batch_size, seq_len = prompt_end_inputs['input_ids'].shape
    
    tokenized_list.append(prompt_end_inputs)
    position_ids2_list.append(torch.ones(seq_len, dtype=torch.long).to(model.device) * cur_pos2)
    
    position_ids2 = torch.cat(position_ids2_list).to(model.device)
    position_ids2 = position_ids2.unsqueeze(0) #1 x seq len
    
    input_ids = torch.cat([t["input_ids"] for t in tokenized_list], dim=1).to(model.device)
    attention_mask = torch.cat([t["attention_mask"] for t in tokenized_list], dim=1).to(model.device)
    
    full_prompt = f"Summarize the following government report:\n\n{doc}\n\nSummary:"

    with torch.no_grad():
        if dual == "block":
            output = model.generate(
                #**inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1
            )
        else:
            output = model.generate(
                #**inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids_2 = position_ids2,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1
            )
    
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_part = full_output[len(full_prompt):].strip()
    print("GENERATED: ", generated_part)
    return generated_part

def test_govreport(model, tokenizer, args, dual="dual"):
    # Load dataset
    print("[INFO] Loading GovReport dataset...")
    #dataset = load_dataset("ccdv/govreport-summarization", split="test")
    dataset_struct = load_dataset("launch/gov_report", name="structure", split="test")
    dataset_plain = load_dataset("launch/gov_report", name="plain_text", split="test")
    print(dataset_plain)
    print(dataset_struct)

    # Import ROUGE
    print("[INFO] Loading ROUGE metric...")
    rouge = evaluate.load("rouge")

    preds = []
    refs = []

    print(f"[INFO] Running evaluation on {args.n} samples...")
    
    modelsz = ""
    
    if args.model == "Qwen/Qwen3-1.7B":
        modelsz="17B"
    elif args.model == "Qwen/Qwen3-0.6B":
        modelsz="06B"
    elif args.model == "Qwen/Qwen3-4B":
        modelsz="4B"
    
    qualitative_file_name = f"{dual}_govreport_generation_results_{args.rope}_{modelsz}.txt"
    quantitative_file_name = f"{dual}_govreport_results_{args.rope}.txt"
    
    with open(qualitative_file_name, 'a') as f:
        f.write("\n=== EVAL CONFIGS ===\n")
        f.write(f"Samples: {args.n}\n")
        f.write(f"Quantization: {args.quant}\n")
        f.write(f"Long summarization: {args.long}\n")

        for i in tqdm(range(args.n)):
    
            doc = dataset_struct[i]["document_sections"]
            ref = dataset_plain[i]["summary"]
            
            pred = summarize(model, tokenizer, doc, dual=dual)
            preds.append(pred)
            refs.append(ref)
            
            if i <= 10:
                f.write("\n##################\n")
                f.write(f"Reference Summary \n {ref}\n")
                f.write(f"Model Summary \n {pred} \n")
    
        print("[INFO] Computing ROUGE...")
        scores = rouge.compute(
            predictions=preds,
            references=refs,
            use_stemmer=True
        )
    
        print("\n=== FINAL RESULTS ===")
        for k, v in scores.items():
            print(f"{k}: {v:.10f}")
    
        save_results_to_file(scores, args, filename=quantitative_file_name)



def main(args):
    # Load model
    model, tokenizer = load_model(args.model, args.quant)

    #Qwen3Model = modeling_qwen3.Qwen3Model
    #Qwen3Model.forward = dual_pos_foward
    #Qwen3ModelForCausalLM = modeling_qwen3.Qwen3ForCausalLM
    #Qwen3ForCausalLM.forward = dual_pos_causal_forward
    
    
    #Qwen3RotaryEmbedding = modeling_qwen3.Qwen3RotaryEmbedding
   # Qwen3RotaryEmbedding.forward = hirope_forward_dual
    
    if args.rope == 'hirope':
        Qwen3RotaryEmbedding = modeling_qwen3.Qwen3RotaryEmbedding
        Qwen3RotaryEmbedding.forward = hirope_forward_dual
        print("Patched HiRoPE 2 as RotaryEmbeddingForward")

        modeling_qwen3.Qwen3Model.forward = Qwen3Model_forward_patch
        modeling_qwen3.Qwen3ForCausalLM.forward = Qwen3Causal_forward_patch
    elif args.rope == 'sumrope':
        Qwen3RotaryEmbedding = modeling_qwen3.Qwen3RotaryEmbedding
        Qwen3RotaryEmbedding.forward = sumrope_forward_dual
        print("Patched SUMROPE 2 as RotaryEmbeddingForward")

        modeling_qwen3.Qwen3Model.forward = Qwen3Model_forward_patch
        modeling_qwen3.Qwen3ForCausalLM.forward = Qwen3Causal_forward_patch
    elif args.rope == 'sumhirope':
        Qwen3RotaryEmbedding = modeling_qwen3.Qwen3RotaryEmbedding
        Qwen3RotaryEmbedding.forward = sumhirope_forward_dual
        print("Patched SUMROPE 2 as RotaryEmbeddingForward")

        modeling_qwen3.Qwen3Model.forward = Qwen3Model_forward_patch
        modeling_qwen3.Qwen3ForCausalLM.forward = Qwen3Causal_forward_patch
        
    #print(model.model.rotary_emb.forward
    # model.Qwen3RotaryEmbedding.forward = hirope_forward
    #print("Patched Qwen3RotaryEmbedding to HiRoPE")

    #test_govreport(model, tokenizer, args, dual="dual")
    test_govreport(model, tokenizer, args, dual="block")
    # test_narrativeqa(model, tokenizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--quant", choices=[None, "8bit", "4bit"], default=None)
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--output", type=str, default="", help="Output filename for results")
    parser.add_argument("--rope", type=str, default="base", help="Rope Embedding to Use")
    args = parser.parse_args()

    main(args)
