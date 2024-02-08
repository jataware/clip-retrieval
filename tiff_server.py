
import os
import numpy as np

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from rio_tiler.profiles import img_profiles
from rio_tiler.io import ImageReader
from rio_tiler.errors import TileOutsideBounds


# # <<
# def _get_shape(src, zz):
#     xx = 0
#     while src.tile_exists(xx, 0, zz):
#         xx += 1
    
#     yy = 0
#     while src.tile_exists(0, yy, zz):
#         yy += 1
    
#     return xx, yy

# _data = {}

# from tqdm import trange
# with ImageReader('cog.tif') as src:
#     _, nrow, ncol, _ = src.bounds
#     for zz in trange(src.minzoom, src.maxzoom + 1):
#         xmax, ymax = _get_shape(src, zz)
#         for xx in range(xmax):
#             for yy in range(ymax):
#                 img = src.tile(xx, yy, zz)
#                 _data[(xx, yy, zz)] = img.render(img_format="PNG", **img_profiles.get("png"))

# # >>

app = FastAPI(
    title="rio-tiler",
    description="A lightweight Cloud Optimized GeoTIFF tile server",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    r"/{z}/{x}/{y}.png",
    responses={
        200: {"content": {"image/png": {}}, "description": "Return an image.",}
    },
    response_class=Response,
    description="Read COG and return a tile",
)
def tile(z: int, x: int, y: int, url: str):
    # try:
    #     content = _data[(x, y, z)]
    #     return Response(content, media_type="image/png")
    # except:
    #     pass

    try:
        with ImageReader(url) as src:
            img = src.tile(x, y, z)
        
        content = img.render(img_format="PNG", **img_profiles.get("png"))
        return Response(content, media_type="image/png")

    except TileOutsideBounds:
        return None
        
    except Exception as e:
        raise(e)

@app.get("/tilejson.json", responses={200: {"description": "Return a tilejson"}})
def tilejson(
    req: Request,
    url: str = Query(..., description="Cloud Optimized GeoTIFF URL."),
):
    """Return TileJSON document for a COG."""
    tile_url = str(req.url_for("tile", z="{z}", x="{x}", y="{y}"))
    tile_url = f"{tile_url}?url={url}"

    with ImageReader(url) as cog:
        return {
            "minZoom"  : cog.minzoom,
            "maxZoom"  : cog.maxzoom,
            "tileSize" : 256,
            "nrow"     : 32,
            "ncol"     : 32,
            "name"     : os.path.basename(url),
            "tiles"    : [tile_url],
        }

# !! need to point to file
@app.post('/labels')
async def labels(req: Request):
    data = await req.json()
    data = np.array(list(data['labels'].values()))
    print(data)
    data = data.reshape(32, 32)
    data = data[::-1]
    print(data, (data != 0).sum())
    print('saving...')
    np.save('labels.npy', data)