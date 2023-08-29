#!/usr/bin/env python

"""
    clip_back.py
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
from dataclasses import dataclass
from typing import Callable, Any, List

from flask import Flask, request
from flask_restful import Resource, Api

import faiss
import torch

from clip_retrieval.load_clip import load_clip, get_tokenizer

class Health(Resource):
    def get(self):
        return "OK!"


def download_image(url):
    """Download an image from a url and return a byte stream"""
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    urllib_context = ssl.create_default_context()
    urllib_context.set_alpn_protocols(["http/1.1"])

    with urllib.request.urlopen(urllib_request, timeout=10, context=urllib_context) as r:
        img_stream = io.BytesIO(r.read())
    
    return img_stream


# class MetadataService(Resource):
#     def __init__(self, clip_resource):
#         super().__init__()
#         self.clip_resource = clip_resource

#     def post(self):
#         json_data = request.get_json(force=True)
#         ids       = json_data["ids"]
#         if len(ids) == 0:
#             return []
        
#         metadata_provider  = self.metadata_provider
#         metas              = metadata_provider.get(ids, self.columns_to_return)        
#         return [{"id": item_id, "metadata": meta_to_dict(meta)} for item_id, meta in zip(ids, metas)]


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class KNNService(Resource):
    """the knn service provides nearest neighbors given text or image"""

    def __init__(self, 
        device,
        model,
        preprocess,
        tokenizer,
        metadata_provider,
        index,
        columns_to_return,
    ):
        super().__init__()

        self.device             = device
        self.model              = model
        self.preprocess         = preprocess
        self.tokenizer          = tokenizer
        self.metadata_provider  = metadata_provider
        self.index              = index
        self.columns_to_return  = columns_to_return
    
    # def map_to_metadata(self, indices, distances, n_imgs, metadata_provider, columns_to_return):

    #     results = []
    #     metas   = metadata_provider.get(indices[:n_imgs], columns_to_return)
        
    #     for key, (d, i) in enumerate(zip(distances, indices)):
    #         output = {}
    #         meta   = None if key + 1 > len(metas) else metas[key]
    #         convert_metadata_to_base64(meta)
    #         if meta is not None:
    #             output.update(meta_to_dict(meta))
            
    #         output["id"]         = i.item()
    #         output["similarity"] = d.item()
    #         results.append(output)

    #     return results

    def query(self, q_text, q_img, q_emb, n_imgs, n_mids):
        
        if n_mids is None:
            n_mids = n_imgs
        
        # --
        # Compute query
        
        if q_text is not None and q_text != "":
            inp = self.tokenizer([q_text]).to(self.device)
            q   = self.model.encode_text(inp)
            q  /= q.norm(dim=-1, keepdim=True)
            q   = q.cpu().to(torch.float32).detach().numpy()
        
        elif q_img is not None:
            inp = Image.open(q_img)
            inp = self.preprocess(inp).unsqueeze(0).to(self.device)
            q  = self.model.encode_image(inp)
            q /= q.norm(dim=-1, keepdim=True)
            q  = q.cpu().to(torch.float32).detach().numpy()
        
        elif q_emb is not None:
            q = np.array(q_emb).astype(np.float32)[None]
        
        # --
        # KNN Search
        
        D, I, E = self.index.search_and_reconstruct(q, n_mids)
        D, I, E = D[0], I[0], E[0]
        
        # drop missing entries
        sel     = I != -1
        D, I, E = D[sel], I[sel], E[sel]
        
        # !! deduplicate?
        # !! safety filters, etc?
        
        if len(D) == 0:
            return []
        
        # --
        # Hydrate w/ metadata
        
        # results = self.map_to_metadata(I, D, n_imgs, self.metadata_provider, self.columns_to_return)
        # return results
        
        return [{"index" : int(_index), "distance" : float(_distance)} for _index, _distance in zip(I, D)]
    
    def post(self):
        req = request.get_json(force=True)
        
        # --
        # Load image, if appropriate
        
        if req.get("img", None) is not None:
            q_img = BytesIO(base64.b64decode(req['img']))
        elif req.get("img_url", None) is not None:
            q_img = download_image(req['img_url'])
        else:
            q_img = None
        
        
        return self.query(
            q_text = req.get("text", None),
            q_img  = q_img,
            q_emb  = req.get("emb",  None),
            n_imgs = req["n_imgs"],
            n_mids = req.get("n_mids", None),
        )



def load_index(path, enable_faiss_memory_mapping):
    if enable_faiss_memory_mapping:
        if os.path.isdir(path):
            return faiss.read_index(path + "/populated.index", faiss.IO_FLAG_ONDISK_SAME_DIR)
        else:
            return faiss.read_index(path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    
    else:
        return faiss.read_index(path)



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


# def load_metadata_provider(
#     index_folder, index, columns_to_return, use_arrow
# ):
#     """load the metadata provider"""
#     parquet_folder    = index_folder + "/metadata"
#     mmap_folder       = parquet_folder
#     metadata_provider = ArrowMetadataProvider(mmap_folder)
    
#     return metadata_provider



def clip_back(
    index_folder,
    clip_model,
    
    columns_to_return              = ["url", "caption"],
    
    use_jit                        = True,
    enable_faiss_memory_mapping    = True,
    use_arrow                      = True,
    
    port                           = 1234,
):
    
    # --
    # Load CLIP model
    
    device            = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip(clip_model, use_jit=use_jit, device=device)
    tokenizer         = get_tokenizer(clip_model)
    index             = load_index(index_folder + "/image.index", enable_faiss_memory_mapping)

    # --
    # Load index
    
    metadata_provider = None # load_metadata_provider(
    #     index_folder,
    #     index,
    #     columns_to_return,
    #     use_arrow,
    # )
    
    # --
    # CLIPResource
    
    params = dict(
        device            = device,
        model             = model,
        preprocess        = preprocess,
        tokenizer         = tokenizer,
        metadata_provider = metadata_provider,
        index             = index,
        columns_to_return = columns_to_return,
    )

    # --
    # Run App
    
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(Health,          "/health")
    # api.add_resource(MetadataService, "/metadata",     resource_class_kwargs={"clip_resource" : clip_resource})
    api.add_resource(KNNService,      "/knn-service",  resource_class_kwargs=params)
        
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    fire.Fire(clip_back)
