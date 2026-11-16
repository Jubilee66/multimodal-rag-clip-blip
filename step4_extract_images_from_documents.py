#=========================================================
# Image
#  ↓
# CLIP
#  ↓
# Embedding
# (Phase1-Phase5)
#=========================================================
import pymupdf
import os

# =====================================
# FOLDERS
# =====================================

DATA_FOLDER = "data"

IMAGE_OUTPUT_FOLDER = os.path.join(
    DATA_FOLDER,
    "extracted_images"
)

os.makedirs(
    IMAGE_OUTPUT_FOLDER,
    exist_ok=True
)

# =====================================
# PROCESS ALL PDFs
# =====================================

total_images = 0

for file_name in os.listdir(DATA_FOLDER):

    if not file_name.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(
        DATA_FOLDER,
        file_name
    )

    print(f"\nProcessing: {file_name}")

    try:

        doc = pymupdf.open(pdf_path)

        image_count = 0

        for page_num in range(len(doc)):

            page = doc.load_page(page_num)

            image_list = page.get_images(
                full=True
            )

            for image_index, img in enumerate(
                image_list
            ):

                xref = img[0]

                base_image = doc.extract_image(
                    xref
                )

                image_bytes = base_image[
                    "image"
                ]

                image_ext = base_image[
                    "ext"
                ]

                image_name = (
                    f"{os.path.splitext(file_name)[0]}"
                    f"_page_{page_num + 1}"
                    f"_img_{image_index}"
                    f".{image_ext}"
                )

                image_path = os.path.join(
                    IMAGE_OUTPUT_FOLDER,
                    image_name
                )

                with open(
                    image_path,
                    "wb"
                ) as image_file:

                    image_file.write(
                        image_bytes
                    )

                image_count += 1
                total_images += 1

        print(
            f"Extracted {image_count} images"
        )

    except Exception as e:

        print(
            f"Error processing "
            f"{file_name}: {e}"
        )

print("\n===================================")
print(f"TOTAL IMAGES: {total_images}")
print("Image extraction complete.")
print("===================================")