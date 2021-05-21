from torch.utils.data import Dataset
import pandas as pd
import torch

batch_size = 256

class CustomDPRDataset(Dataset):
    def __init__(self, p_type="train"):
        p_type = 'dpr_custom/' + p_type + '.csv'
        dataset_df = pd.read_csv(p_type)
        self.dataset = []
        all_other_docs = []

        for query, highest_doc, lowest_doc in zip(dataset_df['query'], dataset_df['gold_doc'], dataset_df['neg_sample']):
            curr_tup = [query, highest_doc, lowest_doc]
            self.dataset.append(curr_tup)

        for row_i in range(len(self.dataset)):
            all_other_docs = []
            for i in range(batch_size-2):
                o_col = 'o' + str(i+1)
                curr_other_doc = dataset_df[o_col].iloc[row_i]
                all_other_docs.append(curr_other_doc)
            self.dataset[row_i].append(all_other_docs)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[idx]