import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import gradio as gr

# Load model and tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#Use for english-sanskrit translations
model_name = "ai4bharat/indictrans2-en-indic-1B"
#Use for sanskrit-english translations
# model_name = "ai4bharat/indictrans2-indic-en-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# Translation function
def translate_text(input_sentences):
    if isinstance(input_sentences, str):
        input_sentences = [input_sentences]  # Convert to list if single sentence

    # Preprocess input
    batch = ip.preprocess_batch(input_sentences, src_lang="eng_Latn", tgt_lang="sans_Deva")

    # Set device
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

    # Generate translations
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,  # Adjust this as needed
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode generated tokens
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess translations
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    # Return first translation (in case there are multiple sentences)
    return translations[0]

# Gradio interface
def gradio_translate(input_sentence,):
    return translate_text(input_sentence)

# Define Gradio UI

interface = gr.Interface(
    fn=gradio_translate,
    inputs=[
        gr.Textbox(lines=2, label="Input Sentence", placeholder="Enter sentence to translate"),
    ],
    outputs="text",
    title="Indic Language Translator",
    description="Translate sentences between English and various Indic languages using the IndicTrans model.",
)

# Launch Gradio interface
interface.launch()
