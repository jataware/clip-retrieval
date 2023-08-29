#!/usr/bin/env python

"""
    clip_back.py
    
    Dropped:
        - Safety / violence / aesthetic
        - Various performance flags that we weren't using
        - De-duplication
"""

import os
import io
import ssl
import fire
import base64
import urllib
import numpy as np
from tqdm import tqdm
import pyarrow as pa
from pyarrow.ipc import RecordBatchFileReader

from PIL import Image
from io import BytesIO
from pathlib import Path

from flask import Flask, request
from flask_restful import Resource, Api

import faiss
import torch
from torch import autocast, nn
import open_clip
torch.backends.cuda.matmul.allow_tf32 = True

# --
# Model

class OpenClipWrapper(nn.Module):
    def __init__(self, model, preprocess, tokenizer, device):
        super().__init__()
        self.model      = model
        self.preprocess = preprocess
        self.tokenizer  = tokenizer
        self.device     = torch.device(device=device)
        
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def encode_image(self, image):
        if self.device.type == "cpu":
            return self.model.encode_image(image)
        
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.model.encode_image(image)

    def encode_text(self, text):
        if self.device.type == "cpu":
            return self.model.encode_text(text)
        
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.model.encode_text(text)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _warmup(model, warmup_batch_size, device):
    fake_img     = Image.new("RGB", (224, 224), color="red")
    fake_text    = ["fake"] * warmup_batch_size
    image_tensor = torch.cat([torch.unsqueeze(model.preprocess(fake_img), 0)] * warmup_batch_size).to(device)
    text_tokens  = model.tokenizer(fake_text).to(device)
    for _ in range(2):
        with torch.no_grad():
            model.encode_image(image_tensor)
            model.encode_text(text_tokens)


def load_open_clip(clip_model, use_jit=True, warmup_batch_size=1, clip_cache_path=None, device=None):
    clip_model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = open_clip.get_tokenizer(clip_model)
    
    pretrained           = dict(open_clip.list_pretrained())
    checkpoint           = pretrained[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=checkpoint, device=device, jit=use_jit, cache_dir=clip_cache_path)
    model                = OpenClipWrapper(model, preprocess, tokenizer, device=device)
    model                = model.to(device=device)
    
    
    _warmup(model, warmup_batch_size, device)
    return model


# --
# Helpers

def download_img(url):
    headers        = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"}
    urllib_request = urllib.request.Request(url, data=None, headers=headers,)
    urllib_context = ssl.create_default_context()
    urllib_context.set_alpn_protocols(["http/1.1"])

    with urllib.request.urlopen(urllib_request, timeout=10, context=urllib_context) as r:
        img_stream = io.BytesIO(r.read())
    
    return img_stream


# --
# KNN-Search

class KNNService(Resource):
    """the knn service provides nearest neighbors given text or image"""

    def __init__(self, model, index, meta, device):
        super().__init__()

        self.model  = model
        self.index  = index
        self.meta   = meta
        self.device = device

    def post(self):
        req = request.get_json(force=True)
        
        # --
        # Parse req + load image, if appropriate
        
        q_text      = req.get("text", None)
        q_emb       = req.get("emb",  None)
        n_imgs      = req["n_imgs"]
        n_meta      = req.get("n_meta", n_imgs)
        return_embs = req.get("return_embs", False)
        
        # fetch image (maybe)
        if req.get("img", None) is not None:
            q_img = BytesIO(base64.b64decode(req['img']))
        elif req.get("img_url", None) is not None:
            q_img = download_img(req['img_url'])
        else:
            q_img = None
        
        # compute query
        if q_text is not None and q_text != "":
            inp = self.model.tokenizer([q_text]).to(self.device)
            q   = self.model.encode_text(inp)
            q  /= q.norm(dim=-1, keepdim=True)
            q   = q.cpu().to(torch.float32).detach().numpy()
        
        elif q_img is not None:
            inp = Image.open(q_img)
            inp = self.model.preprocess(inp).unsqueeze(0).to(self.device)
            q   = self.model.encode_image(inp)
            q /= q.norm(dim=-1, keepdim=True)
            q  = q.cpu().to(torch.float32).detach().numpy()
        
        elif q_emb is not None:
            q = np.array(q_emb).astype(np.float32)[None]
        
        # do search
        D, I, E = self.index.search_and_reconstruct(q, n_imgs) # !! could run multiple at a time
        D, I, E = D[0], I[0], E[0]
        
        # drop missing entries
        sel     = I != -1
        D, I, E = D[sel], I[sel], E[sel]
        
        # !! deduplicate? (if so, remember to normalize E)
        # !! safety filters, etc?
        
        if len(I) == 0: return []
        
        results = [{
            "id"  : int(ii),
            "sim" : float(dd),
            "emb" : [float(xx) for xx in ee] if return_embs else None, # ?? encode this some other way?  base64?
            "url" : None,
        } for dd, ii, ee in zip(D, I, E)]
                
        # Hydrate w/ metadata
        metas = self.meta.get(I[:n_meta])
        for i, meta in enumerate(metas):
            results[i]['url'] = meta['url']
        
        return results
        
        

# --
# Endpoints

class ArrowMetadataProvider:
    def __init__(self, arrow_folder):
        arrow_files = [str(a) for a in sorted(Path(arrow_folder).glob("**/*")) if a.is_file()]
        self.table  = pa.concat_tables([RecordBatchFileReader(pa.memory_map(arrow_file, "r")).read_all() for arrow_file in tqdm(arrow_files)])

    def get(self, ids):
        t = pa.concat_tables([self.table[i : i + 1] for i in ids])
        return t.to_pandas().to_dict("records")

class HydrateService(Resource):
    def __init__(self, meta):
        super().__init__()
        self.meta = meta

    def post(self):
        req = request.get_json(force=True)
        
        ids = req["ids"]
        if len(ids) == 0:
            return []
        
        metas = self.meta.get(ids)
        return [{"id": item_id, "metadata": meta} for item_id, meta in zip(ids, metas)]


class Health(Resource):
    def get(self):
        return "OK!"


def clip_back(index_folder, clip_model):
    assert 'open_clip:' in clip_model
    clip_model = clip_model.replace('open_clip:', '')
    
    # --
    # Load metadata
    
    print('load_meta     : start')
    meta = ArrowMetadataProvider(os.path.join(index_folder, "metadata"))
    print('load_meta     : complete')
    
    # --
    # Load CLIP model
    
    print('load_open_clip: start')
    device = "cuda" if torch.cuda.is_available() else "cpu" # Is this right?  I guess we want CUDA if we have it...
    model  = load_open_clip(clip_model, use_jit=True, device=device)
    print('load_open_clip: complete')
    
    # --
    # Load index
    
    print('read_index    : start')
    index = faiss.read_index(os.path.join(index_folder, "image.index/populated.index"), faiss.IO_FLAG_ONDISK_SAME_DIR)
    print('read_index    : complete')
        
    # --
    # Run App

    params = dict(
        model  = model,
        index  = index,
        meta   = meta,
        device = device,
    )
    
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(Health,          "/health")
    api.add_resource(HydrateService,  "/hydrate",     resource_class_kwargs={"meta" : meta})
    api.add_resource(KNNService,      "/knn-service", resource_class_kwargs=params)
        
    app.run(host="0.0.0.0", port=1234, debug=False)

if __name__ == "__main__":
    fire.Fire(_run)
