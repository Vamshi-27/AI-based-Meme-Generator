import streamlit as st
from PIL import Image
from src.clip_gpt2_utils import ClipGPT2Model

@st.cache_resource
def load_model():
    return ClipGPT2Model(device="cpu")

def main():
    st.title("Meme Caption Generator (CLIP + GPT-2)")
    st.write("Upload an image to generate a meme caption.")

    uploaded_file = st.file_uploader("Choose a meme image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = load_model()

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = model.generate_caption(image)
                st.success(f"Generated Caption: {caption}")

if __name__ == "__main__":
    main()
