import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from torch import nn
from vdjdb_preprocessing.process_tsv import VdjdbTsv
from training.prepare_input_dataset import assign_embeddings
from training.read_embeddings import read_h5py_embeddings, read_vdj_dataset_csv
from training.model_per_seq import NeuralNetwork
from training.utils import predict_class_label_by_majority
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

vdjcdr3_emb_dict = read_h5py_embeddings("jh_data/T5_embeddings/score_thresh1/V-CDR3-J_embedding_per_seq.h5")
epi_emb_dict = read_h5py_embeddings("jh_data/T5_embeddings/score_thresh1/Epitope_embedding_per_seq.h5")

test_vdj_epi_df = read_vdj_dataset_csv("jh_data/jh_trb_mhc1_scorethres1_hs.csv")

print(len(vdjcdr3_emb_dict))
print(len(epi_emb_dict))
print(test_vdj_epi_df.shape[0])
print(test_vdj_epi_df['Class'].value_counts())

pred_dict = {}
batch_size = 500

for model_part in range(1, 21):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model", model_part)

    model = NeuralNetwork()
    model.load_state_dict(torch.load('saved_models/vdjdb_trb_mhc1_scorethres1_hs_per_seq' + str(model_part) + '.pth'))
    model.to(device)

    model.eval()
    with torch.no_grad():
        y_test_pred = list()
        for batch_t in range(0, test_vdj_epi_df.shape[0], batch_size):
            print("Batch", batch_t)
            b_vdj_epi_df_test = test_vdj_epi_df.iloc[batch_t:batch_t + batch_size]
            b_X_test_vdj, _ = assign_embeddings(vdjcdr3_emb_dict, epi_emb_dict, b_vdj_epi_df_test)
            b_X_test_vdj = b_X_test_vdj.to(device)
            b_y_test_pred = model(b_X_test_vdj)
            b_y_test_pred = torch.sigmoid(b_y_test_pred)
            b_y_test_pred = b_y_test_pred.squeeze().cpu().detach().numpy()
            y_test_pred = np.append(y_test_pred, b_y_test_pred)

        y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)

        for idx, pred_label in enumerate(y_test_pred):
            if idx not in pred_dict:
                pred_dict[idx] = list()
                pred_dict[idx].append(pred_label)
            else:
                pred_dict[idx].append(pred_label)

y_pred_majority = predict_class_label_by_majority(pred_dict, majority_thresh=18)
test_vdj_epi_df['pred_label'] = y_pred_majority

tr_b_tsv = VdjdbTsv('db/vdjdb_trb_mhc1_scorethres1_hs.tsv')
epi_df = tr_b_tsv.get_epitope_df()

test_vdj_epi_df = test_vdj_epi_df.merge(epi_df, how='left', on='Epitope')
test_vdj_epi_df.to_csv("jh_data/jh_trb_mhc1_scorethres1_hs_pred_per_seq.csv", index=False)