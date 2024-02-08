import requests
import numpy as np
import pandas as pd
from tifffile import imwrite as tiffwrite

import os
from PIL import Image
from tqdm import tqdm

# --
# Helpers

def do_query(**kwargs):
    out = requests.post("http://0.0.0.0:1234/knn-service", 
        headers={"Content-Type" : "application/json"},
        json=kwargs,
    ).json()
    return pd.DataFrame(out)


df   = do_query(text="the flag of djibouti", n_imgs=1_000, return_embs=True)
df   = df.drop_duplicates('url')
df   = df.reset_index(drop=True)

with open('_urls.txt', 'w') as f:
    print('\n'.join(df.url), file=f)

# ...  img2dataset     --url_list                _urls.txt     --output_folder           _imgs     --thread_count            64     --image_size              256     --resize_only_if_bigger   True     --resize_mode             keep_ratio_largest

dl_meta = pd.read_parquet('_imgs/00000.parquet').sort_values('key')
dl_meta = dl_meta[['url', 'status', 'key']]
df      = pd.merge(df, dl_meta, how='left')
df      = df[df.status == 'success']
assert df.key.notnull().all()

# --

from umap import UMAP
import rasterfairy as rf

from rcode import *
import matplotlib.pyplot as plt

# compute umap
embedding = UMAP(n_neighbors=5, min_dist=0.5, metric='cosine').fit_transform(np.row_stack(df.emb.values))

# project to grid
grid_xy, grid_sz = rf.transformPointCloud2D(embedding)

df['_c'] = grid_xy[:,0].astype(int)
df['_r'] = grid_xy[:,1].astype(int)
df       = df.sort_values(["_r", '_c']).reset_index(drop=True)

D = 128

out  = np.zeros((D * grid_sz[1], D * grid_sz[0], 3), dtype=np.uint8)
mask = np.zeros((grid_sz[1], grid_sz[0]), dtype=bool)
for r, c, key in tqdm(zip(df._r, df._c, df.key), total=df.shape[0]):
    c, r = int(c), int(r)
    
    _img = Image.open(os.path.join('_imgs/00000', key + '.jpg'))
    _img = _img.convert('RGB')
    _img = _img.resize((D, D))
    _img = np.asarray(_img)
    
    out[(D*r):(D*r+D), (D*c):(D*c+D)] = _img
    mask[r,c] = True

tiffwrite('tmp.tif', out)

# gdal_translate -of COG tmp.tif cog.tif
# ... annotations through the app ...

labs       = np.load('labels.npy')
df['_ann'] = labs[:mask.shape[0],:mask.shape[1]][mask]

from sklearn.svm import LinearSVC
X = np.row_stack(df.emb.values)
y = df._ann.values

# train a model ... but need random background samples as well
Xn = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
m = LinearSVC().fit(Xn, y == 1)
m = m.coef_.squeeze()
m = m / np.sqrt((m ** 2).sum())

mm = np.row_stack(df.emb[df._ann == 1].values).mean(axis=0)

z = do_query(emb=list(mm), n_imgs=1000)
z = z[~z.url.isin(df.url)]

print('\n'.join(z.url))