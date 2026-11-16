from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import numpy as np

IMAGE_FOLDER = "data/extracted_images"

model = SentenceTransformer(
    "clip-ViT-B-32"
)

all_embeddings = []

for file in os.listdir(IMAGE_FOLDER):

    path = os.path.join(
        IMAGE_FOLDER,
        file
    )

    try:

        image = Image.open(
            path
        ).convert("RGB")

        embedding = model.encode(
            image
        )

        all_embeddings.append(
            {
                "image": file,
                "embedding": embedding
            }
        )

        print(f"Embedded: {file}")

    except Exception as e:

        print(
            f"Error: {file} -> {e}"
        )

np.save(
    "image_embeddings.npy",
    all_embeddings,
    allow_pickle=True
)

print(
    f"\nStored {len(all_embeddings)} image embeddings."
)


# from sentence_transformers import SentenceTransformer
# from PIL import Image
# import os
# import numpy as np

# IMAGE_FOLDER = "data/extracted_images"

# model = SentenceTransformer(
#     "clip-ViT-B-32"
# )

# all_embeddings = []

# for file in os.listdir(IMAGE_FOLDER):

#     path = os.path.join(
#         IMAGE_FOLDER,
#         file
#     )

#     try:

#         image = Image.open(
#             path
#         ).convert("RGB")

#         embedding = model.encode(
#             image
#         )

#         all_embeddings.append(
#             {
#                 "image": file,
#                 "embedding": embedding
#             }
#         )

#         print(f"Embedded: {file}")

#     except Exception as e:

#         print(
#             f"Error: {file} -> {e}"
#         )

# np.save(
#     "image_embeddings.npy",
#     all_embeddings,
#     allow_pickle=True
# )

# print(
#     f"\nStored {len(all_embeddings)} image embeddings."
# )
# from sentence_transformers import SentenceTransformer
# from PIL import Image
# import os
# import numpy as np

# IMAGE_FOLDER = "data/extracted_images"

# model = SentenceTransformer(
#     "clip-ViT-B-32"
# )

# all_embeddings = []

# for file in os.listdir(IMAGE_FOLDER):

#     path = os.path.join(
#         IMAGE_FOLDER,
#         file
#     )

#     try:

#         image = Image.open(
#             path
#         ).convert("RGB")

#         embedding = model.encode(
#             image
#         )

#         all_embeddings.append(
#             {
#                 "image": file,
#                 "embedding": embedding
#             }
#         )

#         print(f"Embedded: {file}")

#     except Exception as e:

#         print(
#             f"Error: {file} -> {e}"
#         )

# np.save(
#     "image_embeddings.npy",
#     all_embeddings,
#     allow_pickle=True
# )

# print(
#     f"\nStored {len(all_embeddings)} image embeddings."
# )
