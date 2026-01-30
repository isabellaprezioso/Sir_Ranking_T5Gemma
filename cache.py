from transformers import AutoTokenizer, T5EncoderModel

# DECOMMENTA QUELLO CHE VUOI SCARICARE NELLA CACHE
model_name = "google/t5gemma-2b-2b-prefixlm-it" 
# model_name = "google/t5gemma-2-1b-1b"
# model_name = "google/t5gemma-2-270m-270m"

print(f"Inizio download di {model_name} nella cache...")

try:
    AutoTokenizer.from_pretrained(model_name)
    T5EncoderModel.from_pretrained(model_name)
    print("Fatto. Modello e tokenizer sono ora in ~/.cache/huggingface/")
except Exception as e:
    print(f"Errore: {e}. Sei sul nodo di login? Hai internet?")
