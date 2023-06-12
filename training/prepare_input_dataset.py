import numpy as np
import pandas as pd
from numpy import save
from torch.utils.data import Dataset, DataLoader
import torch


def assign_embeddings(vdjcdr3_emb_dict, epi_emb_dict, vdj_epi_df):
    vdj_epi_df = vdj_epi_df.reset_index(drop=True)
    vdj_epi_df['CDR3-V-J'] = vdj_epi_df['CDR3'].map(str) + "#" + vdj_epi_df['V'].map(str) + "#" + vdj_epi_df['J'].map(str)
    vdj_epi_df['vdjcdr3_emb'] = vdj_epi_df.apply(lambda x: np.array(vdjcdr3_emb_dict[x['CDR3-V-J']]), axis=1)
    vdj_epi_df['epi_emb'] = vdj_epi_df.apply(lambda x: np.array(epi_emb_dict[x['Epitope']]), axis=1)
    vdj_epi_df['vdjcdr3_epi_emb'] = vdj_epi_df.apply(lambda x: np.append(x['vdjcdr3_emb'], x['epi_emb']), axis=1)
    vdj_epi_df = vdj_epi_df.drop('vdjcdr3_emb', axis=1)
    vdj_epi_df = vdj_epi_df.drop('epi_emb', axis=1)
    X = np.array(vdj_epi_df['vdjcdr3_epi_emb'].to_list())
    y = np.array(vdj_epi_df['Class'])

    '''for idx, row in vdj_epi_df.iterrows():
        vdjcdr3_emb = vdjcdr3_emb_dict[row['CDR3'] + "#" + row['V'] + "#" + row['J']]
        epi_emb = epi_emb_dict[row['Epitope']]
        vdjcdr3_emb = np.reshape(vdjcdr3_emb, (1, -1))
        epi_emb = np.reshape(epi_emb, (1, -1))
        X_row = np.append(vdjcdr3_emb, epi_emb, axis=1)
        if idx == 0:
            X = X_row
        else:
            X = np.append(X, X_row, axis=0)
        print(X.shape)

    save(X_outfile, X)

    y = vdj_epi_df['Class']
    y = np.reshape(y, (-1, 1))
    save(y_outfile, y)'''

    return [X, y]

def assign_embeddings_per_residue(vdjcdr3_emb_dict, epi_emb_dict, vdj_epi_df):
    vdj_epi_df = vdj_epi_df.reset_index(drop=True)
    vdj_epi_df['CDR3-V-J'] = vdj_epi_df['CDR3'].map(str) + "#" + vdj_epi_df['V'].map(str) + "#" + vdj_epi_df['J'].map(
        str)
    vdj_epi_df['vdjcdr3_emb'] = vdj_epi_df.apply(lambda x: np.array(vdjcdr3_emb_dict[x['CDR3-V-J']]), axis=1)
    vdj_epi_df['epi_emb'] = vdj_epi_df.apply(lambda x: np.array(epi_emb_dict[x['Epitope']]), axis=1)

    X_vdj = np.array(vdj_epi_df['vdjcdr3_emb'].to_list())
    X_epi = np.array(vdj_epi_df['epi_emb'].to_list())
    y = np.array(vdj_epi_df['Class'])



    X_vdj = torch.tensor(X_vdj, dtype=torch.float32)
    X_epi = torch.tensor(X_epi, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    X_vdj = X_vdj.transpose(1, 2).contiguous()
    X_epi = X_epi.transpose(1, 2).contiguous()



    return([X_vdj, X_epi, y])


class Data(Dataset):
    def __init__(self, X_train, y_train):
        # need to convert float64 to float32 else
        # will get the following error
        # RuntimeError: expected scalar type Double but found Float
        self.X = torch.from_numpy(X_train.astype(np.float32))
        # need to convert float64 to Long else
        # will get the following error
        # RuntimeError: expected scalar type Long but found Float
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len