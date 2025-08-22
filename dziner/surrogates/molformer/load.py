import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import numpy as np

class MolFormer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.invalid = []

    def get_embedding(self, smiles):
        encoding = self.tokenizer(smiles["smiles"], padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encoding)

        model_output = outputs.pooler_output

        encoding["embedding"] = model_output

        return encoding


    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                                  trust_remote_code=True)


    def encode(self, smiles_list=[], use_gpu=False, return_tensor=True):
        if type(smiles_list) != list:
            smiles_list = list(smiles_list)
        smiles_df = pd.DataFrame(smiles_list, columns=["smiles"])
        data = Dataset.from_pandas(smiles_df)
        embedding = data.map(self.get_embedding, batched=True, num_proc=1, batch_size=128)
        emb = np.asarray(embedding["embedding"].copy())


        if return_tensor:
            return torch.tensor(emb)
        return pd.DataFrame(emb)
