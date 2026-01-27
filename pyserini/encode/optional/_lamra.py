import yaml
from typing import Any, List
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class LamRABaseEncoder:
    def add_embed_token(self, emb_token="<emb>"):
        emb_tokens = [emb_token]
        num_new_tokens = self.tokenizer.add_tokens(emb_tokens)
        assert len(emb_tokens) == num_new_tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        emb_token_ids = self.tokenizer.convert_tokens_to_ids(emb_tokens)
        self.model.config.emb_token_ids = emb_token_ids

    def get_embed_feature(self, hidden_states, input_ids, embed_index):
        embed_indices = torch.argmax((input_ids == embed_index).int(), dim=1)
        embed_features = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]
        return embed_features

    def qwen2vl_process(self, messages):
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        return inputs

    def _get_summarization_instruction(self, modality: str) -> str:
        if modality == "image,text":
            instruction = "\nSummarize above image and sentence in one word: "
        elif modality == "text":
            instruction = "\nSummarize above sentence in one word: "
        elif modality == "image":
            instruction = "\nSummarize above image in one word: "
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return instruction

    def format_message(
        self,
        modality: str,
        txt: str | None,
        img_path: str | None,
        instruction: str | None
    ) -> List[dict]:
        summarization_instr = self._get_summarization_instruction(modality)
        if txt and instruction:
            combined_text = f"{instruction} {txt}{summarization_instr}"
        elif txt:
            combined_text = f"{txt}{summarization_instr}"
        elif instruction:
            combined_text = f"{instruction}{summarization_instr}"
        else:
            combined_text = f"{summarization_instr}"

        if modality == "image,text": # image and text
            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": combined_text}
                ],
            }, {
                "role": "assistant",
                "content": [{"type": "text", "text": "<emb>."}]
            }]
        elif modality == "text": # text only
            message = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_text}
                ],
            }, {
                "role": "assistant",
                "content": [{"type": "text", "text": "<emb>."}]
            }]
        else:  # image only
            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": combined_text}
                ],
            }, {
                "role": "assistant",
                "content": [{"type": "text", "text": "<emb>."}]
            }]
        return message

class LamRADocumentEncoder(LamRABaseEncoder):
    def __init__(
        self,
        model_name: str,
        device="cuda:0",
        l2_norm=False,
        **kwargs: Any
    ):
        self.device = device
        self.l2_norm = l2_norm
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        self.add_embed_token()  

    def encode(
        self,
        dids: List[int],
        img_paths: List[str | None],
        modalitys: List[str],
        txts: List[str],
        **kwargs: Any
    ):
        messages_batch = []
        
        for img_path, modality, txt in zip(img_paths, modalitys, txts):
            messages_batch.append(self.format_message(modality=modality, txt=txt, img_path=img_path, instruction=None))
        
        inputs = self.qwen2vl_process(messages_batch)
        
        with torch.no_grad():
            hidden_states = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
            embeds = self.get_embed_feature(hidden_states, inputs['input_ids'], self.model.config.emb_token_ids[0])
            
            if self.l2_norm:
                embeds = F.normalize(embeds, dim=-1)
        
        return embeds.cpu().numpy()

class LamRAQueryEncoder(LamRABaseEncoder):
    def __init__(
        self,
        encoder_dir: str,
        device="cuda:0",
        l2_norm=False,
        instruction_config: str = None,
        **kwargs: Any
    ):
        self.device = device
        self.l2_norm = l2_norm
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            encoder_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(encoder_dir)
        self.tokenizer = self.processor.tokenizer
        self.instruction_config = instruction_config
        
        self.add_embed_token()

    def _get_instruction_config(self, instr_file: str = None):
        """This functions downloads all the instruction config files if not already present."""

        import os
        import tarfile
        from pyserini.util import download_url, get_cache_home

        cache_dir = get_cache_home()
        instructions_dir = os.path.join(cache_dir, 'query_instructions')

        if not os.path.exists(instructions_dir):
            query_images_and_instructions_url = "https://huggingface.co/datasets/castorini/prebuilt-indexes-m-beir/resolve/main/mbeir_query_images_and_instructions.tar.gz"
            tar_path = os.path.join(cache_dir, 'mbeir_query_images_and_instructions.tar.gz')

            try:  
                download_url(query_images_and_instructions_url, cache_dir, force=False)
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(cache_dir)
            except Exception as e:
                raise Exception(f"Could not download query images: {e}")

        if instr_file:
            return os.path.join(instructions_dir, instr_file)
        else:
            return None

    def _load_instruction_config(self, instruction_config):
        try:
            with open(instruction_config, "r") as f:
                config = yaml.safe_load(f)
            instruction_file = config.get("instruction_file", None)
            candidate_modality = config.get("candidate_modality", None)
            dataset_id = config.get("dataset_id", None)
            randomize_instructions = config.get("randomize_instructions", False)
            if instruction_file is None or candidate_modality is None or dataset_id is None:
                raise ValueError(
                    "Instruction file, candidate_modality, or dataset_id is missing in the config. Please download the instruction file from https://huggingface.co/datasets/TIGER-Lab/M-BEIR/blob/main/instructions/query_instructions.tsv"
                )
        except Exception as e:
            raise ValueError(f"Error loading instruction config: {e}")

        try:
            import pandas as pd
            df = pd.read_csv(instruction_file, sep="\t")
            filtered = df[df["dataset_id"].astype(int) == int(dataset_id)]
            instructions = filtered.to_dict(orient="records")

            return instructions, candidate_modality, randomize_instructions
        except Exception as e:
            raise ValueError(
                f"Error reading instruction or corpus file: {e}. Please download the instruction file from https://huggingface.co/datasets/TIGER-Lab/M-BEIR/blob/main/instructions/query_instructions.tsv"
            )

    def _get_instruction_prompt(self, instructions, c_modality, q_modality, randomize_instructions) -> str | None:
        import random
        for instruction in instructions:
            if instruction["query_modality"] == q_modality and instruction["cand_modality"] == c_modality:
                if randomize_instructions:
                    prompts = [instruction[k] for k in instruction if k.startswith("prompt_")]
                    return random.choice(prompts) if prompts else None
                else:
                    return instruction["prompt_1"]

    def encode(
        self,
        qid: int,
        query_modality: str,
        query_txt: str = "",
        query_img_path: str = "",
        **kwargs: Any
    ):
        if self.instruction_config is None:
            self.instruction_config = self._get_instruction_config(kwargs.get("instr_file", None))

        instruction = None
        if self.instruction_config:
            instructions, candidate_modality, randomize_instructions = self._load_instruction_config(self.instruction_config)
            instruction = self._get_instruction_prompt(
                instructions=instructions,
                c_modality=candidate_modality,
                q_modality=query_modality,
                randomize_instructions=randomize_instructions,
            )
        inputs = self.qwen2vl_process([self.format_message(modality=query_modality, txt=query_txt, img_path=query_img_path, instruction=instruction)])
        
        with torch.no_grad():
            hidden_states = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
            embeds = self.get_embed_feature(hidden_states, inputs['input_ids'], self.model.config.emb_token_ids[0])
            
            if self.l2_norm:
                embeds = F.normalize(embeds, dim=-1)
        
        return embeds.cpu().numpy().flatten()
