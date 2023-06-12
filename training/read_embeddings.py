import h5py
import numpy as np
import pandas as pd


def read_h5py_embeddings(in_file):
    em_h5 = h5py.File(in_file, "r")
    em_dict = {}
    for em_key in list(em_h5.keys()):
        em_array = em_h5[em_key][()]
        em_dict[em_key] = em_array

    em_h5.close()
    return em_dict

def read_vdj_dataset_csv(in_file):
    vdj_df = pd.read_csv(in_file, header=None)
    vdj_df.columns = ["CDR3", "V", "J", "Epitope", "Class"]
    return vdj_df
