#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import yaml
import random
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset

from pyserini.uniir import (BLIPFeatureFusion, BLIPScoreFusion,
                            CLIPFeatureFusion, CLIPScoreFusion,
                            MBEIRCandidatePoolCollator, format_string,
                            hash_did, hash_qid)

class CustomCorpusDataset(Dataset):
    def __init__(self, batch_info, img_preprocess_fn, **kwargs):
        data = []
        num_records = len(batch_info["img_path"])
        for i in range(num_records):
            record = {
                "did": batch_info["did"][i],
                "img_path": batch_info["img_path"][i],
                "modality": batch_info["modality"][i],
                "txt": batch_info["txt"][i],
            }
            data.append(record)
        self.data = data
        self.img_preprocess_fn = img_preprocess_fn
        self.kwargs = kwargs
      
    def __len__(self):  
        return len(self.data)

    def __getitem__(self, idx):  
        entry = self.data[idx]
        img_path = entry.get("img_path", None)
        if not img_path:
            img = None
        else:
            img = Image.open(img_path).convert("RGB")
            img = self.img_preprocess_fn(img)

        did = entry.get("did", None)
        did = hash_did(did)
        cand_txt = entry.get("txt", "")
        cand_txt = format_string(cand_txt)
        cand_modality = entry.get("modality", None)

        instance = {
            "did": did,
            "txt": cand_txt,
            "img": img,
            "modality": cand_modality,
        }

        return instance


class UniIRDatasetConverter:
    def __init__(self, batch_info, img_preprocess_fn, tokenizer, **kwargs):
        dataset = CustomCorpusDataset(batch_info, img_preprocess_fn, **kwargs)
        batch_size = len(batch_info["img_path"])
        collator = MBEIRCandidatePoolCollator(
            tokenizer=tokenizer, image_size=(224, 224)
        )
        self.data = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    def get_data(self):
        return self.data


MODEL_REGISTRY = {
    "clip_ff": (CLIPFeatureFusion, "CLIP_FF"),
    "clip_sf": (CLIPScoreFusion, "CLIP_SF"),
    "blip_ff": (BLIPFeatureFusion, "BLIP_FF"),
    "blip_sf": (BLIPScoreFusion, "BLIP_SF"),
}


class UniIREncoder(ABC):
    def __init__(self, model_name: str, device="cuda:0", l2_norm=False, **kwargs: Any):
        clip_vision_model = "ViT-L/14" if "large" in model_name else "ViT-B/32"

        model_key = next((key for key in MODEL_REGISTRY if key in model_name), None)
        if not model_key:
            raise ValueError(f"Unsupported model name for UniIR: {model_name}")

        ModelClass, model_dir = MODEL_REGISTRY[model_key]
        model = ModelClass(model_name=clip_vision_model, device=device)

        try:
            checkpoint_path = hf_hub_download(
                repo_id="TIGER-Lab/UniIR",
                filename=f"checkpoint/{model_dir}/{model_name}.pth",
            )
        except Exception as e:
            raise ValueError(
                f"Model checkpoint not found: {e}. Please check the model name or ensure the model is available on Hugging Face Hub: https://huggingface.co/TIGER-Lab/UniIR/tree/main/checkpoint."
            )

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=False)[
                "model"
            ]
        )
        model.eval()
        model = model.to(device)

        self.model = model
        self.img_preprocess_fn = model.get_img_preprocess_fn()
        self.tokenizer = model.get_tokenizer()
        self.device = device
        self.l2_norm = l2_norm

    @abstractmethod
    def encode(self, **kwargs: Any):
        pass


class UniIRCorpusEncoder(UniIREncoder):
    def __init__(self, model_name: str, device="cuda:0", l2_norm=False, **kwargs: Any):
        super().__init__(model_name, device, l2_norm, **kwargs)

    def encode(self, dids, img_paths, modalitys, txts, **kwargs: Any):
        if kwargs.get("fp16", False):
            self.model.half()
        else:
            self.model.float()

        batch_info = {
            "did": dids,
            "img_path": img_paths,
            "modality": modalitys,
            "txt": txts,
        }
        dataset = UniIRDatasetConverter(
            batch_info=batch_info,
            img_preprocess_fn=self.img_preprocess_fn,
            tokenizer=self.tokenizer,
        ).get_data()

        with torch.no_grad():
            batch = next(iter(dataset))
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            corpus_embeddings, _ = self.model.encode_mbeir_batch(batch)
            corpus_embeddings = corpus_embeddings.cpu().numpy()
            if self.l2_norm:
                corpus_embeddings = normalize(corpus_embeddings, axis=1, norm="l2")
            return corpus_embeddings


class CustomQueryDataset(Dataset):
    def __init__(self, query_info, img_preprocess_fn, **kwargs):
        self.query_info = query_info
        self.img_preprocess_fn = img_preprocess_fn
        self.kwargs = kwargs
        
    def __len__(self):
        return len(self.query_info)
        
    def __getitem__(self, idx):
        entry = self.query_info[idx]
          
        query_img_path = entry.get("query_img_path", None)
        if query_img_path:
            img = Image.open(query_img_path).convert("RGB")
            img = self.img_preprocess_fn(img)
        else:
            img = None
          
        query_txt = entry.get("query_txt", "")
        query_txt = format_string(query_txt)
          
        query = {
            "txt": query_txt,
            "img": img
        }
          
        instance = {"query": query}
          
        qid = entry.get("qid", None)
        if qid:
            instance.update({"qid": hash_qid(qid)})
              
        return instance

class UniIRQueryConverter:
    def __init__(self, query_info, img_preprocess_fn, tokenizer, **kwargs):
        dataset = CustomQueryDataset(query_info, img_preprocess_fn, **kwargs)
        collator = MBEIRInferenceOnlyCollator(tokenizer=tokenizer, image_size=(224, 224))
        self.data = DataLoader(dataset, batch_size=1, collate_fn=collator)
  
    def get_data(self):
        return self.data

class UniIRQueryEncoder(UniIREncoder):
    def __init__(self, encoder_dir: str, device='cuda:0', l2_norm=False, instruction_config=None, **kwargs: Any):
        if instruction_config is not None:
            instructions, modality_info, random_instruction = self._load_instruction_config(instruction_config)
            self._instructions = instructions
            self._modality_info = modality_info
            self._random_instruction = random_instruction
        super().__init__(encoder_dir, device, l2_norm, **kwargs)

    def _load_instruction_config(self, instruction_config):
        try:
            with open(instruction_config, 'r') as f:
                config = yaml.safe_load(f)
            instruction_file = config.get('instruction_file', None)
            corpus_file = config.get('corpus_file', None)
            dataset_id = config.get('dataset_id', None)
            random_instruction = config.get('random_instruction', False)
            if not instruction_file or not corpus_file or not dataset_id:
                raise ValueError("Instruction file or corpus file not specified in the config.")
        except Exception as e:
            raise ValueError(f"Error loading instruction config: {e}")

        try:
            df = pd.read_csv(instruction_file, sep='\t')
            filtered = df[df['dataset_id'].astype(int) == int(dataset_id)]
            instructions = filtered.to_dict(orient="records")

            modality_info = {}
            with open(corpus_file, "r") as f:
                for line in f:
                    corpus = json.loads(line)
                    modality_info[corpus['did']] = corpus['modality']

            return instructions, modality_info, random_instruction
        except Exception as e:
            raise ValueError(f"Error reading instruction or corpus file: {e}")

    def _get_instruction_prompt(self, q_modality, c_modality) -> str:
        instructions = self._instructions
        for instruction in instructions:
            if instruction['query_modality'] == q_modality and instruction['cand_modality'] == c_modality:
                if self._random_instruction:
                    prompts = [instruction[k] for k in instruction if k.startswith('prompt_')]
                    return random.choice(prompts)
                else:
                    return instruction['prompt_1']

    def encode(self, qid, query_txt, query_img_path, query_modality, pos_cand_list, **kwargs: Any):
        if kwargs.get('fp16', False):
            self.model.half()
        else:
            self.model.float()

        cand_modality = self._modality_info.get(pos_cand_list[0], 'text') if hasattr(self, '_modality_info') else 'text'
        

        if hasattr(self, '_instructions'):
            prompt = self._get_instruction_prompt(
                q_modality=query_modality,
                c_modality=cand_modality
            )
            query_txt = f"{prompt} {query_txt}" if prompt else query_txt

        query_info = {
            "qid": qid,
            "query_txt": query_txt,
            "query_img_path": query_img_path if query_img_path else None,
            "query_modality": query_modality,
            "candidate_modality": cand_modality,
        }
        query_info = [query_info]  # Wrap in a list to match the dataset format

        query_dataset = UniIRQueryConverter(
            query_info=query_info,
            img_preprocess_fn=self.img_preprocess_fn,
            tokenizer=self.tokenizer,
        ).get_data()

        with torch.no_grad():
            batch = next(iter(query_dataset))
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            query_embeddings, _ = self.model.encode_mbeir_batch(batch)
            query_embeddings = query_embeddings.cpu().numpy()
            if self.l2_norm:
                query_embeddings = normalize(query_embeddings, axis=1, norm='l2')
            return query_embeddings
