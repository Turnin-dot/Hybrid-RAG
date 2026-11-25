"""BGE-base-en-v1.5 embedding and Cross-Encoder reranking wrappers.

Both are lazy-loaded and reused, so the model is loaded once per process.
Vectors are L2-normalized and match the 768-dim model version; the index is
therefore bound to this exact model revision.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import os
from functools import cached_property

import numpy as np


def file_sha256(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def model_snapshot(name):
    path=Path(name)
    if path.is_dir():return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name,cache_dir=str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache'/'huggingface')))/'hub'),local_files_only=True))


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(str(self.snapshot), device=self.device,local_files_only=True)
        return self._model

    @cached_property
    def snapshot(self):return model_snapshot(self.model_name)

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.snapshot,local_files_only=True,trust_remote_code=False)

    @cached_property
    def identity(self):
        names=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','model.safetensors','pytorch_model.bin','modules.json','sentence_bert_config.json','config_sentence_transformers.json','1_Pooling/config.json']
        return dict(model=self.model_name,revision=self.snapshot.name,dimension=self.dim,normalize=True,
                    files_sha256={n:file_sha256(self.snapshot/n) for n in names if (self.snapshot/n).exists()})

    def validate_lengths(self,texts):
        for i in range(0,len(texts),64):
            lengths=list(map(len,self.tokenizer(texts[i:i+64],add_special_tokens=True,truncation=False,padding=False)['input_ids']))
            if any(n>512 for n in lengths):raise ValueError('Embedding input exceeds 512 Tokens; silent truncation is forbidden')

    def encode(self, texts: list[str], batch_size: int = 128, normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        self.validate_lengths(texts)
        model = self._load()
        return model.encode(
            texts, batch_size=batch_size, normalize_embeddings=normalize,
            convert_to_numpy=True, show_progress_bar=False,
        )

    @property
    def dim(self) -> int:
        return 768

    @property
    def revision(self) -> str:
        # Reported to the experiment manifest so index provenance is explicit.
        return f"{self.model_name}@{self.snapshot.name}"


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(str(model_snapshot(self.model_name)), device=self.device,local_files_only=True)
        return self._model

    @cached_property
    def identity(self):
        p=model_snapshot(self.model_name)
        return dict(model=self.model_name,revision=p.name,pair_overflow_policy='window_max',max_length=512,window_overlap=32,
            files_sha256={f.name:file_sha256(f) for f in p.iterdir() if f.name in {'model.safetensors','config.json','tokenizer.json','tokenizer_config.json'}})

    def scores(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        if not pairs:
            return []
        model = self._load()
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        expanded=[];groups=[];tok=model.tokenizer
        for query,text in pairs:
            length=len(tok(query,text,add_special_tokens=True,truncation=False)['input_ids'])
            if length<=512:windows=[text]
            else:
                budget=512-len(tok(query,add_special_tokens=True,truncation=False)['input_ids'])-8
                if budget<=32:raise ValueError('Reranker query leaves no safe document budget')
                windows=RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tok,chunk_size=budget,chunk_overlap=32,strip_whitespace=False).split_text(text)
            if not windows or any(len(tok(query,w,add_special_tokens=True,truncation=False)['input_ids'])>512 for w in windows):raise ValueError('Reranker pair exceeds budget; refusing implicit truncation')
            groups.append((len(expanded),len(expanded)+len(windows)))
            expanded.extend((query,w) for w in windows)
        out=model.predict(expanded,batch_size=batch_size,show_progress_bar=False)
        self.last_window_counts=[b-a for a,b in groups]
        return [max(float(x) for x in out[a:b]) for a,b in groups]

    def rerank(self, query: str, texts: list[str], batch_size: int = 32) -> list[tuple[int, float]]:
        scores = self.scores([(query, t) for t in texts], batch_size=batch_size)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order]
