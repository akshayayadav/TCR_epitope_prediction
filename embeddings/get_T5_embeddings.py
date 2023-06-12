import torch
import re
import h5py
from transformers import T5EncoderModel, T5Tokenizer


class T5model:
    def __init__(self):
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print("Using device: {}".format(self.device))
        self.model_dir = "embeddings/T5/"
        self.transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
        if self.model_dir is not None:
            print("##########################")
            print("Loading cached model from: {}".format(self.model_dir))
            print("##########################")
        self.model = T5EncoderModel.from_pretrained(self.transformer_link, cache_dir=self.model_dir)
        # self.model.full() if self.device == 'cpu' else self.model.half()  # only cast to full-precision if no GPU is available

        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.vocab = T5Tokenizer.from_pretrained(self.transformer_link, do_lower_case=False)

    def calculate_embeddings_vdj(self, vdj_dict, batch_size=50, per_seq=0):
        print('V-CDR3-J', 'embedding')
        if per_seq == 0:
            out_file = h5py.File("embeddings/T5_embeddings/" + 'V-CDR3-J' + "_embedding.h5", 'w')
        else:
            out_file = h5py.File("embeddings/T5_embeddings/" + 'V-CDR3-J' + "_embedding_per_seq.h5", 'w')
        vcdr3j_keys = list(vdj_dict.keys())
        for dict_index in range(0, len(vcdr3j_keys), batch_size):
            start_index = dict_index
            print('Starting index', start_index)
            end_index = dict_index + (batch_size)
            if end_index>len(vcdr3j_keys):
                end_index = len(vcdr3j_keys)

            vcdr3j_keys_batch = vcdr3j_keys[start_index:end_index]

            vdj_seq_list = [vdj_dict[vdj_id]['V-CDR3-J'] for vdj_id in vcdr3j_keys_batch]
            #vdj_seq_list_len = [len(seq) for seq in vdj_seq_list]
            embedding_repr = self.get_embeddings(vdj_seq_list, self.vocab, self.model, self.device)

            for vcdr3j_idx, vcdr3j_id in enumerate(vcdr3j_keys_batch):
                v_seq_len = len(vdj_dict[vcdr3j_id]['V_seq'])
                cdr3_len = len(vdj_dict[vcdr3j_id]['CDR3'])
                emb = embedding_repr.last_hidden_state[vcdr3j_idx, v_seq_len:v_seq_len + cdr3_len]
                print(cdr3_len, emb.shape)
                if per_seq == 1:
                    emb = emb.mean(dim=0)
                emb = emb.detach().cpu().numpy().squeeze()
                out_file.create_dataset(vcdr3j_id, data=emb)

        out_file.close()

    def calculate_embeddings_epitope(self, epitope_dict, batch_size=50, per_seq=0):
        print('Epitope', 'embedding')
        if per_seq == 0:
            out_file = h5py.File("embeddings/T5_embeddings/" + 'Epitope' + "_embedding.h5", 'w')
        else:
            out_file = h5py.File("embeddings/T5_embeddings/" + 'Epitope' + "_embedding_per_seq.h5", 'w')
        epitope_seqs = list(epitope_dict.keys())
        for seq_idx in range(0, len(epitope_seqs), batch_size):
            start_index = seq_idx
            print('Starting index', start_index)
            end_index = seq_idx + batch_size
            if end_index>len(epitope_seqs):
                end_index = len(epitope_seqs)

            epitope_seqs_batch = epitope_seqs[start_index: end_index]
            #epitope_seqs_batch_len = [len(seq) for seq in epitope_seqs_batch]
            embedding_repr = self.get_embeddings(epitope_seqs_batch, self.vocab, self.model, self.device)

            for epi_idx, epi_seq in enumerate(epitope_seqs_batch):
                emb = embedding_repr.last_hidden_state[epi_idx, 0:len(epi_seq)]
                print(len(epi_seq), emb.shape)
                if per_seq == 1:
                    emb = emb.mean(dim=0)
                emb = emb.detach().cpu().numpy().squeeze()
                out_file.create_dataset(epi_seq, data=emb)

        out_file.close()

    @staticmethod
    def get_embeddings(seq_list, vocab, model, device):
        seq_list = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq_list]
        token_encoding = vocab.batch_encode_plus(seq_list, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        with torch.no_grad():
            embedding_repr = model(input_ids, attention_mask=attention_mask)

        return embedding_repr


    '''def calculate_embeddings_vdj(self, vdj_df, type = 'V-CDR3-J', batch_size=50):
        print(type, 'embedding')
        out_file = h5py.File("embeddings/T5_embeddings/" + type + "_embedding.h5", 'w')
        for df_index in range(0, vdj_df.shape[0], batch_size):
            start_index = df_index
            end_index = df_index + batch_size
            print('Starting batch index', start_index)
            batch_df = vdj_df.iloc[start_index:end_index, :]
            if type == 'V-CDR3-J':
                seq_list = batch_df['V-CDR3-J'].to_list()
                seq_len_list = [len(seq) for seq in seq_list]
            elif type == 'Epitope':
                seq_list = batch_df['Epitope'].to_list()
                seq_len_list = [len(seq) for seq in seq_list]
            seq_list = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq_list]

            token_encoding = self.vocab.batch_encode_plus(seq_list, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(self.device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)

            with torch.no_grad():
                embedding_repr = self.model(input_ids, attention_mask=attention_mask)

            seq_idx = 0
            for index, row in batch_df.iterrows():
                if type == 'V-CDR3-J':
                    v_seq_len = len(row['V_seq'])
                    cdr3_len = len(row['CDR3'])
                    emb = embedding_repr.last_hidden_state[seq_idx, v_seq_len:v_seq_len + cdr3_len]
                elif type == 'Epitope':
                    emb = embedding_repr.last_hidden_state[seq_idx, 0:seq_len_list[seq_idx]]
                seq_idx += 1
                emb = emb.detach().cpu().numpy().squeeze()
                out_file.create_dataset(str(index), data=emb)
        out_file.close()

    def test_calculate_embeddings(self, seq_list):
        seq_len_list = [len(seq) for seq in seq_list]
        seq_list = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq_list]
        token_encoding = self.vocab.batch_encode_plus(seq_list, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(token_encoding['input_ids']).to(self.device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
        with torch.no_grad():
            embedding_repr = self.model(input_ids, attention_mask=attention_mask)
        emb = embedding_repr.last_hidden_state[0, 0:seq_len_list[0]]
        emb = emb.detach().cpu().numpy().squeeze()
        print(emb.shape)

    def test2_calculate_embeddings(self, sequence_examples=["PRTEINO", "SEQWENCE"]):
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
        ids = self.vocab.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
            emb = embedding_repr.last_hidden_state
        emb_0 = embedding_repr.last_hidden_state[0, :7]  # shape (7 x 1024)
        print(f"Shape of per-residue embedding of first sequences: {emb_0.shape}")
        emb_1 = embedding_repr.last_hidden_state[1, :8]  # shape (8 x 1024)
        print(f"Shape of per-residue embedding of second sequences: {emb_1.shape}")
        print(emb.shape)'''
