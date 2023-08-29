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
from PIL import Image
from io import BytesIO

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

    def __init__(self, model, index, meta, cols, device):
        super().__init__()

        self.model = model
        self.index = index
        
        self.meta  = meta
        self.cols  = cols
        
        
        self.device = device
    
    # def map_to_metadata(self, indices, distances, n_imgs):

    #     results = []
    #     metas   = self.meta.get(indices[:n_imgs], self.cols)
        
    #     for key, (d, i) in enumerate(zip(distances, indices)):
    #         output = {}
    #         meta   = None if key + 1 > len(metas) else metas[key]
    
    #         if meta is not None:
    #             output.update(meta_to_dict(meta))
            
    #         output["id"]         = i.item()
    #         output["similarity"] = d.item()
    #         results.append(output)

    #     return results

    def post(self):
        req = request.get_json(force=True)
        
        # --
        # Parse req + load image, if appropriate
        
        q_text = req.get("text", None)
        q_emb  = req.get("emb",  None)
        n_imgs = req["n_imgs"]
        n_mids = req.get("n_mids", n_imgs)
        
        if req.get("img", None) is not None:
            q_img = BytesIO(base64.b64decode(req['img']))
        elif req.get("img_url", None) is not None:
            q_img = download_img(req['img_url'])
        else:
            q_img = None

        # --
        # Compute query
        
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
        
        # --
        # KNN Search
        
        D, I, E = self.index.search_and_reconstruct(q, n_mids) # !! could run multiple at a time
        D, I, E = D[0], I[0], E[0]
        
        # drop missing entries
        sel     = I != -1
        D, I, E = D[sel], I[sel], E[sel]

        # --
        # Post-filter
                
        # !! deduplicate? (if so, remember to normalize E)
        # !! safety filters, etc?
        
        if len(D) == 0:
            return []
        
        # --
        # Hydrate w/ metadata
        
        # results = self.map_to_metadata(I, D, n_imgs)
        # return results
        
        return [{"index" : int(_index), "distance" : float(_distance)} for _index, _distance in zip(I, D)]
        
        

# --
# Endpoints

# def meta_to_dict(meta):
#     output = {}
#     for k, v in meta.items():
#         if isinstance(v, bytes):
#             v = v.decode()
#         elif type(v).__module__ == np.__name__:
#             v = v.item()
#         output[k] = v
    
#     return output

# class ArrowMetadataProvider:
#     """The arrow metadata provider provides metadata from contiguous ids using arrow"""

#     def __init__(self, arrow_folder):
#         arrow_files = [str(a) for a in sorted(Path(arrow_folder).glob("**/*")) if a.is_file()]
#         self.table  = pa.concat_tables([pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, "r")).read_all() for arrow_file in arrow_files])

#     def get(self, ids, cols=None):
#         """implement the get method from the arrow metadata provide, get metadata from ids"""
#         if cols is None:
#             cols = self.table.schema.names
#         else:
#             cols = list(set(self.table.schema.names) & set(cols))
        
#         t = pa.concat_tables([self.table[i : i + 1] for i in ids])
#         return t.select(cols).to_pandas().to_dict("records")

# class HydrateService(Resource):
#     def __init__(self, meta, cols, **kwargs):
#         super().__init__()
#         self.meta = meta
#         self.cols = cols

#     def post(self):
#         req = request.get_json(force=True)
#         ids = req["ids"]
#         if len(ids) == 0:
#             return []
        
#         metas = self.meta.get(ids, self.cols)        
#         return [{"id": item_id, "metadata": meta_to_dict(meta)} for item_id, meta in zip(ids, metas)]

class Health(Resource):
    def get(self):
        return "OK!"


def clip_back(index_folder, clip_model):
    assert 'open_clip:' in clip_model
    clip_model = clip_model.replace('open_clip:', '')
    
    # --
    # Load CLIP model
    
    device = "cuda" if torch.cuda.is_available() else "cpu" # Is this right?
    model  = load_open_clip(clip_model, use_jit=True, device=device)
    
    # --
    # Load index
    
    index = faiss.read_index(os.path.join(index_folder, "image.index/populated.index"), faiss.IO_FLAG_ONDISK_SAME_DIR)
    
    # --
    # Load metadata

    # <<
    # parquet_folder    = index_folder + "/metadata"
    # mmap_folder       = parquet_folder
    # meta = ArrowMetadataProvider(mmap_folder)
    # --
    meta = None
    # >>
    
    # --
    # CLIPResource
    
    params = dict(
        model = model,
        index  = index,
        
        meta   = meta,
        cols   = ["url", "caption"],
        
        device = device,
    )

    # --
    # Run App
    
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(Health,          "/health")
    api.add_resource(HydrateService,  "/hydrate",     resource_class_kwargs=params)
    api.add_resource(KNNService,      "/knn-service", resource_class_kwargs=params)
        
    app.run(host="0.0.0.0", port=1234, debug=False)

if __name__ == "__main__":
    fire.Fire(_run)
