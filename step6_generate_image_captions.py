import os
import json

from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

IMAGE_FOLDER = "data/extracted_images"

print("Loading BLIP...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

captions = []

for file in os.listdir(IMAGE_FOLDER):

    path = os.path.join(
        IMAGE_FOLDER,
        file
    )

    try:

        image = Image.open(
            path
        ).convert("RGB")

        inputs = processor(
            image,
            return_tensors="pt"
        )

        output = model.generate(
            **inputs,
            max_new_tokens=50
        )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        captions.append(
            {
                "image": file,
                "caption": caption
            }
        )

        print(
            f"{file} -> {caption}"
        )

    except Exception as e:

        print(
            f"Error: {file} -> {e}"
        )

with open(
    "image_captions.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        captions,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    "\nimage_captions.json created"
)


# import os
# import json

# from PIL import Image

# from transformers import (
#     BlipProcessor,
#     BlipForConditionalGeneration
# )

# IMAGE_FOLDER = "data/extracted_images"

# print("Loading BLIP...")

# processor = BlipProcessor.from_pretrained(
#     "Salesforce/blip-image-captioning-base"
# )

# model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base"
# )

# captions = []

# for file in os.listdir(IMAGE_FOLDER):

#     path = os.path.join(
#         IMAGE_FOLDER,
#         file
#     )

#     try:

#         image = Image.open(
#             path
#         ).convert("RGB")

#         inputs = processor(
#             image,
#             return_tensors="pt"
#         )

#         output = model.generate(
#             **inputs,
#             max_new_tokens=50
#         )

#         caption = processor.decode(
#             output[0],
#             skip_special_tokens=True
#         )

#         captions.append(
#             {
#                 "image": file,
#                 "caption": caption
#             }
#         )

#         print(f"{file}")
#         print(f"Caption: {caption}")
#         print("-" * 50)

#     except Exception as e:

#         print(
#             f"Error processing {file}: {e}"
#         )

# with open(
#     "image_captions.json",
#     "w",
#     encoding="utf-8"
# ) as f:

#     json.dump(
#         captions,
#         f,
#         indent=4,
#         ensure_ascii=False
#     )

# print("\nCaptions saved.")