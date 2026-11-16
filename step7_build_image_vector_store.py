import chromadb
import numpy as np
import json

client = chromadb.PersistentClient(
    path="./chroma_db_img"
)

collection = client.get_or_create_collection(
    name="image_collection"
)

data = np.load(
    "image_embeddings.npy",
    allow_pickle=True
)

with open(
    "image_captions.json",
    "r",
    encoding="utf-8"
) as f:

    captions = json.load(f)

caption_lookup = {
    item["image"]: item["caption"]
    for item in captions
}

for idx, item in enumerate(data):

    image_name = item["image"]

    caption = caption_lookup.get(
        image_name,
        "No caption available"
    )

    collection.add(
        ids=[
            f"img_{idx}"
        ],
        embeddings=[
            item["embedding"].tolist()
        ],
        metadatas=[
            {
                "image": image_name,
                "caption": caption
            }
        ]
    )

    print(
        f"Stored: {image_name}"
    )

print(
    "\nImage collection updated successfully."
)

# import chromadb
# import numpy as np

# client = chromadb.PersistentClient(
#     path="./chroma_db"
# )

# collection = client.get_or_create_collection(
#     "image_collection"
# )

# data = np.load(
#     "image_embeddings.npy",
#     allow_pickle=True
# )

# for idx, item in enumerate(data):

#     collection.add(
#         ids=[str(idx)],
#         embeddings=[item["embedding"].tolist()],
#         metadatas=[
#             {
#                 "image": item["image"]
#             }
#         ]
#     )

# print("Images stored")