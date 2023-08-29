
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
bash Anaconda3-2023.07-2-Linux-x86_64.sh
rm Anaconda3-2023.07-2-Linux-x86_64.sh

sudo apt update
sudo apt install -y aria2
sudo apt install -y build-essential gfortran cmake pkg-config libopenblas-dev

# --

git clone git@github.com:jataware/clip-retrieval
cd clip-retrieval

# --

conda create -y -n lion_env python=3.9
conda activate lion_env

pip install huggingface_hub clip-retrieval
pip install clip-retrieval
pip uninstall clip-retrieval && pip install -e .

# --
# Get download URLS

cd data

echo "" > _index_urls.txt
URL="https://huggingface.co/datasets/laion/laion5b-h14-index/resolve/main/index-parts"
for i in {00..79}; do 
    echo "$URL/$i.index"       >> _index_urls.txt
    echo -e "\tout=$i.parquet" >> _index_urls.txt
done

echo "" > _meta_multi.txt
URL="https://huggingface.co/datasets/laion/laion2b-multi-vit-h-14-embeddings/resolve/main/metadata"
for i in {0000..2268}; do 
    echo "$URL/metadata_$i.parquet" >> _meta_multi.txt
    echo -e "\tout=$i.parquet"      >> _meta_multi.txt
done
# missing 2267 and 2268 - max is 2266

echo "" > _meta_en.txt
URL="https://huggingface.co/datasets/laion/laion2b-en-vit-h-14-embeddings/resolve/main/metadata"
for i in {0000..2313}; do 
    echo "$URL/metadata_$i.parquet" >> _meta_en.txt
    echo -e "\tout=$i.parquet"      >> _meta_en.txt
done

echo "" > _meta_nolang.txt
URL="https://huggingface.co/datasets/laion/laion1b-nolang-vit-h-14-embeddings/resolve/main/metadata"
for i in {0000..1273}; do 
    echo "$URL/metadata_$i.parquet" >> _meta_nolang.txt
    echo -e "\tout=$i.parquet"      >> _meta_nolang.txt
done


# --
# indedx

mkdir -p index-parts && cd index-parts
aria2c                         \
    -j8                        \
    --deferred-input           \
    --conditional-get=true     \
    --auto-file-renaming=false \
    -i ../_index_urls.txt
cd ..

clip-retrieval index_combiner --input_folder "index-parts" --output_folder "combined-indices"


mkdir -p en-embeddings && cd en-embeddings
aria2c                         \
    -j8                        \
    --deferred-input           \
    --conditional-get=true     \
    --auto-file-renaming=false \
    -i ../_meta_en.txt
cd ..

mkdir -p multi-embeddings && cd multi-embeddings
aria2c                         \
    -j8                        \
    --deferred-input           \
    --conditional-get=true     \
    --auto-file-renaming=false \
    -i ../_meta_multi.txt
cd ..

mkdir -p nolang-embeddings && cd nolang-embeddings
aria2c                         \
    -j8                        \
    --deferred-input           \
    --conditional-get=true     \
    --auto-file-renaming=false \
    -i ../_meta_nolang.txt
cd ..

# --

mkdir ./Laion5B_H14 && mkdir ./Laion5B_H14/metadata && mkdir ./Laion5B_H14/image.index
mv combined-indices/* ./Laion5B_H14/image.index/

clip-retrieval back --index_folder data/Laion5B_H14 --clip_model open_clip:ViT-H-14

curl -XPOST http://0.0.0.0:1234/knn-service -H 'Content-Type: application/json' -d '{"text" : "cat", "n_imgs" : 10}'
