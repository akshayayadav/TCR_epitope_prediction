import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from torch import nn
from training.prepare_input_dataset import assign_embeddings_per_residue
from training.read_embeddings import read_h5py_embeddings, read_vdj_dataset_csv
from training.model_per_seq import NeuralNetwork
from sklearn.metrics import precision_recall_fscore_support
from training.model import CNN_TCR_EPI, CNN_EPI_TEST, CNN_VDJ_TEST
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

vdjcdr3_emb_dict = read_h5py_embeddings("embeddings/T5_embeddings/score_thresh1/V-CDR3-J_embedding.h5")
epi_emb_dict = read_h5py_embeddings("embeddings/T5_embeddings/score_thresh1/Epitope_embedding.h5")

test_vdj_epi_df = read_vdj_dataset_csv("db/vdjdb_trb_mhc1_scorethres1/vdjdb_trb_mhc1_scorethres1_hs_pos_neg_test.csv")

print(len(vdjcdr3_emb_dict))
print(len(epi_emb_dict))
print(test_vdj_epi_df.shape[0])
print(test_vdj_epi_df['Class'].value_counts())

pred_dict = {}

for model_part in range(1, 21):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNN_TCR_EPI(1024)
    model.load_state_dict(torch.load('saved_models/vdjdb_trb_mhc1_scorethres1_hs_per_residue'+str(model_part)+'.pth'))
    model.to(device)

    model.eval()

    X_test_vdj, X_test_epi, y_test = assign_embeddings_per_residue(vdjcdr3_emb_dict, epi_emb_dict, test_vdj_epi_df)
    X_test_vdj, X_test_epi, y_test = X_test_vdj.to(device), X_test_epi.to(device), y_test.to(device)
    y_test_pred = model(X_test_vdj, X_test_epi)
    y_test_pred = torch.sigmoid(y_test_pred)

    y_test_pred = y_test_pred.squeeze().cpu().detach().numpy()
    y_test = y_test.squeeze().cpu().detach().numpy()

    y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)

    for idx, pred_label in enumerate(y_test_pred):
        if idx not in pred_dict:
            pred_dict[idx] = list()
            pred_dict[idx].append(pred_label)
        else:
            pred_dict[idx].append(pred_label)

y_pred_majority = list()
for idx in pred_dict:
    preds = pred_dict[idx]
    if preds.count(1) > 18:
        y_pred_majority.append(1)
    else:
        y_pred_majority.append(0)

results_eval = precision_recall_fscore_support(test_vdj_epi_df['Class'].to_list(), y_pred_majority)
print(results_eval)