from torch.utils.data import Dataset
import pandas as pd
import torch

class CustomDPRDataset(Dataset):
    def __init__(self, p_type="train"):
        p_type = 'dpr_custom/' + p_type + '.csv'
        dataset_df = pd.read_csv(p_type)
        self.dataset = []
        all_other_docs = []
        for docs0, docs1, docs2, docs3, docs4, docs5, docs6, docs7, docs8, docs9, docs10, docs11, docs12, docs13 in zip(dataset_df['o0'], dataset_df['o1'], dataset_df['o2'], dataset_df['o3'], dataset_df['o4'], dataset_df['o5'], dataset_df['o6'], dataset_df['o7'], dataset_df['o8'], dataset_df['o9'], dataset_df['o10'], dataset_df['o11'], dataset_df['o12'], dataset_df['o13']):
            docs = [docs0, docs1, docs2, docs3, docs4, docs5, docs6, docs7, docs8, docs9, docs10, docs11, docs12, docs13]
            all_other_docs.append(docs)
        for query, highest_doc, lowest_doc, other_docs in zip(dataset_df['query'], dataset_df['highest_doc'], dataset_df['lowest_doc'], all_other_docs):
            curr_tup = (query, highest_doc, lowest_doc, other_docs)
            self.dataset.append(curr_tup)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[idx]