from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
import faiss
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pyserini.uniir import (BLIPFeatureFusion, BLIPScoreFusion,
                            CLIPFeatureFusion, CLIPScoreFusion,
                            MBEIRCandidatePoolCollator, generate_embeds_and_ids_for_dataset_with_gather,
                            format_string, hash_did)


class CustomCorpusDataset(Dataset):
    def __init__(self, batch_info, img_preprocess_fn, **kwargs):
        data = []
        num_records = len(batch_info["did"])
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
        model.float()
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

    def encode(
        self,
        dids: List[int],
        img_paths: Optional[List[str]] = None,
        modalitys: Optional[List[str]] = None,
        txts: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        use_fp16 = kwargs.get("fp16", False)

        batch_len = len(dids)
        batch_info = {
            "did": dids,
            "img_path": img_paths if img_paths else [None] * batch_len,
            "modality": modalitys if modalitys else ["text"] * batch_len,
            "txt": txts if txts else [""] * batch_len,
        }
        dataloader = UniIRDatasetConverter(
            batch_info=batch_info,
            img_preprocess_fn=self.img_preprocess_fn,
            tokenizer=self.tokenizer,
        ).get_data()

        corpus_embeddings, _ = generate_embeds_and_ids_for_dataset_with_gather(  
            self.model,  
            dataloader,  
            device=self.device,  
            use_fp16=use_fp16,  
        )  

        if self.l2_norm:
            corpus_embeddings = corpus_embeddings.astype('float32')
            faiss.normalize_L2(corpus_embeddings)

        return corpus_embeddings
