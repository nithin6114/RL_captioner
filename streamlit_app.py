import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import torch
import time
from PIL import Image
from io import BytesIO
#----------------------captioner imports----------
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from indicprocessor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image
#-------------------------------------------------

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
img_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
text_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device_type)

max_caption_length = 16
beam_search_count = 4
generation_params = {"max_length": max_caption_length, "num_beams": beam_search_count}

def generate_captions(image, language):
    processed_images = []
    
    processed_images.append(image)

    img_pixels = img_processor(images=processed_images, return_tensors="pt").pixel_values
    img_pixels = img_pixels.to(device_type)

    generated_ids = caption_model.generate(img_pixels, **generation_params)

    captions = text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]
    caption=captions[0]
    print(caption,"############################################################################################3")
    #-------------------------------------convert caption to target language-------------------------------------
    indic_processor = IndicProcessor(inference=True)
    tokenizer_trans = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
    model_trans = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

    if language=="hi":
        processed_input = indic_processor.preprocess_batch(captions, src_lang="eng_Latn", tgt_lang="hin_Deva")
        tokenized_input = tokenizer_trans(processed_input, padding="longest", truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            translation_output = model_trans.generate(**tokenized_input, num_beams=5, num_return_sequences=1, max_length=256)

        with tokenizer_trans.as_target_tokenizer():
            decoded_output = tokenizer_trans.batch_decode(translation_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        final_output = indic_processor.postprocess_batch(decoded_output, lang="hin_Deva")
        final_output = final_output[0]
    elif language=="te":
        processed_input = indic_processor.preprocess_batch(captions, src_lang="eng_Latn", tgt_lang="tel_Telu")
        tokenized_input = tokenizer_trans(processed_input, padding="longest", truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            translation_output = model_trans.generate(**tokenized_input, num_beams=5, num_return_sequences=1, max_length=256)

        with tokenizer_trans.as_target_tokenizer():
            decoded_output = tokenizer_trans.batch_decode(translation_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        final_output = indic_processor.postprocess_batch(decoded_output, lang="tel_Telu")
        final_output = final_output[0]
    else:
        final_output = caption

    return caption, final_output

st.title("Image Captioning with Text-to-Audio")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Choose the translation language:")
    language = st.selectbox("Select language:", ["Select", "english", "telugu", "hindi"])
    lang = None
    if language=="english":
        lang="en"
    elif language=="telugu":
        lang="te"
    elif language=="hindi":
        lang="hi"

    if language!="Select":
        progress_text = st.empty()  # Placeholder for status text
        progress_bar = st.progress(0)
        progress_text.write("Generating caption...")
        for percent in range(0, 101, 10):  # Simulate progress in steps
            time.sleep(1.5)  # Simulate work being done
            progress_bar.progress(percent)
        # st.write("Generating caption...")
        caption, final_output = generate_captions(image, lang)
        progress_text.empty()
        progress_bar.empty()
        # caption = "पृष्ठभूमि में एक पहाड़ और एक बड़े नीले आकाश के साथ एक नदी का दृश्य"
        st.write(f"**English Caption**: {caption}")
        st.write(f"**Selected Language Caption**: {final_output}")

        # Text-to-Audio Conversion
        st.write("Convert the caption to audio:")
        # language = st.selectbox("Select language:", ["en", "es", "fr", "de", "hi"])
        
        if st.button("Generate Audio"):
            if caption.strip():
                try:
                    # Generate audio using gTTS
                    tts = gTTS(text=final_output, lang=lang)
                    
                    # Save to a BytesIO object
                    audio_file = BytesIO()
                    tts.write_to_fp(audio_file)
                    audio_file.seek(0)

                    # Play audio and provide download option
                    st.audio(audio_file, format="audio/mp3")
                    st.download_button(
                        "Download Audio",
                        audio_file,
                        file_name="caption_audio.mp3",
                        mime="audio/mp3",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Caption is empty. Please upload an image first.")