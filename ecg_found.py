from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Edoardo-BS/hubert-ecg-small",
    trust_remote_code=True
)