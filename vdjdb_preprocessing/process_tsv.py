import re
import pandas as pd


class VdjdbTsv:
    def __init__(self, filename):
        self.filename = filename
        self.read_vdj_tsv()

    def read_vdj_tsv(self):
        vdj_df = pd.read_csv(self.filename, sep="\t", header=0)
        self.vdj_df = vdj_df

    def get_vdj_and_epitope_dict(self, tr_bv_dict, tr_bj_dict):
        vdj_dict = {}
        epitope_dict = {}
        for idx, row in self.vdj_df.iterrows():
            vdj_key = row['CDR3'] + "#" + row['V'] + "#" + row['J']
            if not (vdj_key in vdj_dict):
                vdj_dict[vdj_key] = {}
                vdj_dict[vdj_key]['CDR3'] = row['CDR3']
                vdj_dict[vdj_key]['V_seq'] = tr_bv_dict[row['V']]
                vdj_dict[vdj_key]['J_seq'] = tr_bj_dict[row['J']]
                vdj_dict[vdj_key]['V-CDR3-J'] = tr_bv_dict[row['V']] + row['CDR3'] + tr_bj_dict[row['J']]

            epitope_dict[row['Epitope']] = vdj_key

        return [vdj_dict, epitope_dict]

    def assign_v_j_sequences_and_concatenate(self, tr_bv_dict, tr_bj_dict):
        vdj_df = pd.read_csv(self.filename, sep="\t", header=0)
        vdj_df['V_seq'] = vdj_df.apply(lambda x: tr_bv_dict[x['V']], axis=1)
        vdj_df['J_seq'] = vdj_df.apply(lambda x: tr_bj_dict[x['J']], axis=1)
        vdj_df['V-CDR3-J'] = vdj_df['V_seq'].map(str)+vdj_df['CDR3'].map(str)+vdj_df['J_seq']

        return vdj_df

    def get_epitope_df(self):
        epi_df = self.vdj_df[['Epitope', 'Epitope gene', 'Epitope species']]
        epi_df = epi_df.drop_duplicates(subset=['Epitope'], ignore_index=True)
        return epi_df

