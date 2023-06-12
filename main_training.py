import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from training.prepare_input_dataset import assign_embeddings_per_residue
from training.read_embeddings import read_h5py_embeddings, read_vdj_dataset_csv
from training.model import CNN_TCR_EPI, CNN_EPI_TEST, CNN_VDJ_TEST

vdjcdr3_emb_dict = read_h5py_embeddings("embeddings/T5_embeddings/score_thresh1/V-CDR3-J_embedding.h5")
epi_emb_dict = read_h5py_embeddings("embeddings/T5_embeddings/score_thresh1/Epitope_embedding.h5")
vdj_epi_df = read_vdj_dataset_csv("db/vdjdb_trb_mhc1_scorethres1_hs_pos_neg.csv")

print(len(vdjcdr3_emb_dict))
print(len(epi_emb_dict))
print(vdj_epi_df.shape[0])
print(vdj_epi_df['Class'].value_counts())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

vdj_epi_df_train, vdj_epi_df_test, _, _ = train_test_split(vdj_epi_df, vdj_epi_df['Class'], test_size=0.25,
                                                    random_state=42, stratify=vdj_epi_df['Class'])

vdj_epi_df_train = vdj_epi_df_train.reset_index(drop=True)
vdj_epi_df_test = vdj_epi_df_test.reset_index(drop=True)

#hyper parameters
learning_rate = 0.01
epochs = 100
# Model , Optimizer, Loss
model = CNN_TCR_EPI(1024)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
class_weights = torch.Tensor([300]).float()
class_weights = class_weights.to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.squeeze())
#loss_fn = nn.BCEWithLogitsLoss()

batch_size = 1
batch_i = 0

for e_i in range(0, epochs):
    for batch_i in range(0, vdj_epi_df_train.shape[0], batch_size):
        b_vdj_epi_df_train = vdj_epi_df_train.iloc[batch_i:batch_i+batch_size]
        b_X_train_vdj, b_X_train_epi, b_y_train = assign_embeddings_per_residue(vdjcdr3_emb_dict, epi_emb_dict, b_vdj_epi_df_train)

        b_X_train_vdj, b_X_train_epi, b_y_train = b_X_train_vdj.to(device), b_X_train_epi.to(device), b_y_train.to(device)

        model.train()
        optimizer.zero_grad()
        b_y_train_pred = model(b_X_train_vdj, b_X_train_epi)

        loss = loss_fn(b_y_train_pred.squeeze(), b_y_train.squeeze())
        loss.backward()
        optimizer.step()
        if batch_i % 10000 == 0:
            print("Epoch", e_i, "Batch", batch_i, "Loss", loss.item(), "Pred", torch.sigmoid(b_y_train_pred), "True", b_y_train)
    model.eval()
    with torch.no_grad():
        y_test_pred = list()
        y_test = list()
        for batch_j in range(0, vdj_epi_df_test.shape[0], batch_size):
            if batch_j % 1000 == 0:
                print("**Testing**", batch_j)
            b_vdj_epi_df_test = vdj_epi_df_test.iloc[batch_j:batch_j + batch_size]
            b_X_test_vdj, b_X_test_epi, b_y_test = assign_embeddings_per_residue(vdjcdr3_emb_dict, epi_emb_dict, b_vdj_epi_df_test)

            b_X_test_vdj, b_X_test_epi, b_y_test = b_X_test_vdj.to(device), b_X_test_epi.to(device), b_y_test.to(device)
            b_y_test_pred = model(b_X_test_vdj, b_X_test_epi)
            b_y_test_pred = torch.sigmoid(b_y_test_pred)

            b_y_test_pred = b_y_test_pred.squeeze().cpu().detach().numpy()
            b_y_test = b_y_test.squeeze().cpu().detach().numpy()

            y_test_pred = np.append(y_test_pred, b_y_test_pred)
            y_test = np.append(y_test, b_y_test)

        y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)
        results_eval = precision_recall_fscore_support(y_test, y_test_pred)
        print()
        print("Epoch", e_i, "**Testing**", results_eval)
        print()
torch.save(model.state_dict(), "saved_models/vdjdb_trb_mhc1_scorethres1_hs.pth")