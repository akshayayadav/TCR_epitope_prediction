import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from torch import nn

from training.prepare_input_dataset import assign_embeddings
from training.read_embeddings import read_h5py_embeddings, read_vdj_dataset_csv
from training.model_per_seq import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


vdjcdr3_emb_dict = read_h5py_embeddings("embeddings/T5_embeddings/score_thresh1/V-CDR3-J_embedding_per_seq.h5")
epi_emb_dict = read_h5py_embeddings("embeddings/T5_embeddings/score_thresh1/Epitope_embedding_per_seq.h5")

results_dump = open("results_dump_per_seq.txt", "a")

for neg_part in range(1, 22):
    dataset_file = "vdjdb_trb_mhc1_scorethres1_hs_pos_neg"+str(neg_part)+".csv"
    modelfile = "vdjdb_trb_mhc1_scorethres1_hs_per_seq"+str(neg_part)+".pth"

    vdj_epi_df = read_vdj_dataset_csv("db/vdjdb_trb_mhc1_scorethres1/"+dataset_file)

    print(len(vdjcdr3_emb_dict))
    print(len(epi_emb_dict))
    print(vdj_epi_df.shape[0])
    print(vdj_epi_df['Class'].value_counts())

    vdj_epi_df_train, vdj_epi_df_test, _, _ = train_test_split(vdj_epi_df, vdj_epi_df['Class'], test_size=0.25, stratify=vdj_epi_df['Class'])

    vdj_epi_df_train = vdj_epi_df_train.reset_index(drop=True)
    vdj_epi_df_test = vdj_epi_df_test.reset_index(drop=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #hyper parameters
    learning_rate = 0.01
    epochs = 10
    # Model , Optimizer, Loss
    model = NeuralNetwork()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_weights = torch.Tensor([5]).float()
    class_weights = class_weights.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    #loss_fn = nn.BCEWithLogitsLoss()
    batch_size = 200
    for e_i in range(0, epochs):
        for batch_i in range(0, vdj_epi_df_train.shape[0], batch_size):
            b_vdj_epi_df_train = vdj_epi_df_train.iloc[batch_i:batch_i+batch_size]
            b_vdj_epi_X_train, b_vdj_epi_y_train = assign_embeddings(vdjcdr3_emb_dict, epi_emb_dict, b_vdj_epi_df_train)


            #rus = RandomOverSampler(sampling_strategy=0.2)
            #b_vdj_epi_X_train_sple, b_vdj_epi_y_train_sple = rus.fit_resample(b_vdj_epi_X_train, b_vdj_epi_y_train)

            b_vdj_epi_X_train = torch.tensor(b_vdj_epi_X_train, dtype=torch.float32)
            b_vdj_epi_y_train = torch.tensor(b_vdj_epi_y_train, dtype=torch.float32)

            b_vdj_epi_X_train, b_vdj_epi_y_train = b_vdj_epi_X_train.to(device), b_vdj_epi_y_train.to(device)

            model.train()
            optimizer.zero_grad()
            b_vdj_epi_y_train_pred = model(b_vdj_epi_X_train)

            loss = loss_fn(b_vdj_epi_y_train_pred.squeeze(), b_vdj_epi_y_train)
            loss.backward()
            optimizer.step()
            #y_pred = b_vdj_epi_y_train_pred.squeeze().cpu().detach().numpy()
            #y_pred = np.where(y_pred >= 0.5, 1, 0)

            #y_true = b_vdj_epi_y_train.squeeze().cpu().detach().numpy()
            #results_train = precision_recall_fscore_support(y_true, y_pred)

            #print("Epoch", e_i, "Training", results_train, "Loss", loss.item())
            print("Epoch", e_i, "Loss", loss.item())

        model.eval()
        with torch.no_grad():
            vdj_epi_X_test, vdj_epi_y_test = assign_embeddings(vdjcdr3_emb_dict, epi_emb_dict, vdj_epi_df_test)

            vdj_epi_X_test = torch.tensor(vdj_epi_X_test, dtype=torch.float32)
            vdj_epi_y_test = torch.tensor(vdj_epi_y_test, dtype=torch.float32)

            vdj_epi_X_test, vdj_epi_y_test = vdj_epi_X_test.to(device), vdj_epi_y_test.to(device)
            vdj_epi_y_test_pred = model(vdj_epi_X_test)

            vdj_epi_y_test_pred = torch.sigmoid(vdj_epi_y_test_pred)
            y_pred_test = vdj_epi_y_test_pred.squeeze().cpu().detach().numpy()
            y_pred_test = np.where(y_pred_test >= 0.5, 1, 0)

            y_true_test = vdj_epi_y_test.squeeze().cpu().detach().numpy()
            results_eval = precision_recall_fscore_support(y_true_test, y_pred_test)
            print("Epoch", e_i, "**Testing**", results_eval)
            print()
            if e_i == epochs-1:
                results_dump.write(dataset_file+"\n" + "Epoch"+ str(e_i) + "**Testing**" + str(results_eval)+"\n\n")
            #break
        #break
    torch.save(model.state_dict(), "saved_models/"+modelfile)
results_dump.close()
