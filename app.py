#--------------------------------------------

# Document RAG                 ✅
# Image Search                 ✅
# Multimodal Retrieval         ✅
# ChromaDB                     ✅
# CLIP                         ✅
# BLIP                         ✅
# Ollama                       ✅
# Streamlit                    ✅
#--------------------------------------------

import streamlit as st
import os
# from EXTRA_code.step5_Better_Query7 import ask_rag
# from RAG.RAG_2_MultimodalRAG.step9_multimodal_rag import ask_rag
from step9_multimodal_rag import ask_rag

st.set_page_config(
    page_title="My RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 My RAG Assistant")
st.caption(
    "Ask questions from your PDFs, DOCX, Excel and Images"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input(
    "Ask your documents..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # answer = ask_rag(question)

    # with st.chat_message("assistant"):
    #     st.write(answer)

    # st.session_state.messages.append(
    #     {
    #         "role": "assistant",
    #         "content": answer
    #     }
    # )
    result = ask_rag(question)

    with st.chat_message("assistant"):

        st.write(
            result["answer"]
        )

        if result["images"]:

            st.subheader(
                "📷 Retrieved Images"
            )
            for img in result[
                "images"]:

                if os.path.exists(
                    img["path"]):

                    st.image(
                        img["path"],
                        caption=img["name"],
                         width=350#----to adjust image--- 
                        # use_container_width=True
                    )

                else:

                    st.warning(
                        f"Image not found: {img['name']}"
                    )
            # for img in result[
            #     "images"
            # ]:

            #     st.write(
            #         f" {img}"
            #     )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"]
        }
    )

#----TO RUN----
# cd .\RAG\RAG_2_MultimodalRAG\
# .\.venv\Scripts\Activate.ps  OR  ..\..\.venv\Scripts\Activate.ps1
# streamlit run .\__pycache__\app.py
# (inside __pycache__)
# streamlit run app.py

#-----------IMPROVEMENTS ON RAG_2-----------------
# ===Phase 1 - Add Image Captioning===
# EARLIER:Python Image->OCR->Text
# Upgrade:Python Image->OCR->Caption Model
# ===Phase 2 - Add CLIP Embeddings===
# ===Phase 3 - Dual Retrieval===
# ===Phase 4 - Return Actual Images===
# ===Phase 5 - Vision LLM (Big Upgrade)===

#-----Feature Now----



