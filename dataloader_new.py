import torch.utils.data as data
import torch
from utils import integer_label_protein
from mol_graph import sdf_to_graphs

class DTIDataset(data.Dataset):

    def __init__(self, list_IDs, df):
        self.list_IDs = list_IDs
        self.df = df

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        if index >= len(self.df):
            print(f"Error: Index {index} is out of bounds for DataFrame with length {len(self.df)}")
            return None

        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['sdf']
        v_d = sdf_to_graphs(v_d)

        v_p = self.df.iloc[index]['sequence']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]["label"]

        return v_d, torch.tensor(v_p, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
