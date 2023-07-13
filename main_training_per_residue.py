import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from training.prepare_input_dataset import assign_embeddings_per_residue
from training.read_embeddings import read_h5py_embeddings, read_vdj_dataset_csv
from training.model import CNN_TCR_EPI, CNN_EPI_TEST, CNN_VDJ_TEST
from training.utils import get_train_test_split, get_simple_train_test_split, get_positive_class_weight
from sklearn.exceptions import UndefinedMetricWarning
import copy
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

vdjcdr3_emb_dict = read_h5py_embeddings("embeddings/T5_embeddings/score_thresh1/V-CDR3-J_embedding.h5")
epi_emb_dict = read_h5py_embeddings("embeddings/T5_embeddings/score_thresh1/Epitope_embedding.h5")

for neg_part in range(1, 21):
    dataset_file = "vdjdb_trb_mhc1_scorethres1_hs_pos_neg" + str(neg_part) + ".csv"
    modelfile = "vdjdb_trb_mhc1_scorethres1_hs_per_residue" + str(neg_part) + ".pth"

    vdj_epi_df = read_vdj_dataset_csv("db/vdjdb_trb_mhc1_scorethres1/" + dataset_file)

    print("Number of unique TCRs ",len(vdjcdr3_emb_dict))
    print("Number of unique epitopes ", len(epi_emb_dict))

    vdj_epi_df_train, vdj_epi_df_test = get_simple_train_test_split(vdj_epi_df, test_size=0.25)
    print("Training data shape ", vdj_epi_df_train['Class'].value_counts())
    print("Validation data shape ", vdj_epi_df_test['Class'].value_counts())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #hyper parameters
    learning_rate = 0.01
    epochs = 200
    # Model , Optimizer, Loss
    model = CNN_TCR_EPI(1024)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pos_weight = get_positive_class_weight(vdj_epi_df)
    pos_weight = pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.squeeze())
    #loss_fn = nn.BCEWithLogitsLoss()
    batch_size = 500
    train_batch_len = 0
    val_batch_len = 0
    min_val_loss = 999
    no_improve_epoch_tol = 3
    no_improve_epoch_count = 0
    val_loss_decrease_thresh = 0.01


    for e_i in range(0, epochs):
        epoch_train_loss = 0
        for batch_i in range(0, vdj_epi_df_train.shape[0], batch_size):
            #print("Training batch ", batch_i)
            train_batch_len += 1
            b_vdj_epi_df_train = vdj_epi_df_train.iloc[batch_i:batch_i+batch_size]
            b_X_train_vdj, b_X_train_epi, b_y_train = assign_embeddings_per_residue(vdjcdr3_emb_dict, epi_emb_dict,b_vdj_epi_df_train)
            b_X_train_vdj, b_X_train_epi, b_y_train = b_X_train_vdj.to(device), b_X_train_epi.to(device), b_y_train.to(device)

            model.train()
            optimizer.zero_grad()
            b_y_train_pred = model(b_X_train_vdj, b_X_train_epi)
            loss = loss_fn(b_y_train_pred.squeeze(), b_y_train)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        epoch_train_loss = epoch_train_loss / train_batch_len


        model.eval()
        with torch.no_grad():
            y_test_pred = list()
            y_test = list()
            epoch_val_loss = 0
            for batch_t in range(0, vdj_epi_df_test.shape[0], batch_size):
                #print("Validation batch ", batch_t)
                val_batch_len += 1
                b_vdj_epi_df_test = vdj_epi_df_test.iloc[batch_t:batch_t+batch_size]
                b_X_test_vdj, b_X_test_epi, b_y_test = assign_embeddings_per_residue(vdjcdr3_emb_dict, epi_emb_dict, b_vdj_epi_df_test)

                b_X_test_vdj, b_X_test_epi, b_y_test = b_X_test_vdj.to(device), b_X_test_epi.to(device), b_y_test.to(device)

                b_y_test_pred = model(b_X_test_vdj, b_X_test_epi)
                val_loss = loss_fn(b_y_test_pred.squeeze(), b_y_test)

                b_y_test_pred = torch.sigmoid(b_y_test_pred)

                b_y_test_pred = b_y_test_pred.squeeze().cpu().detach().numpy()
                b_y_test = b_y_test.squeeze().cpu().detach().numpy()

                y_test_pred = np.append(y_test_pred, b_y_test_pred)
                y_test = np.append(y_test, b_y_test)

                epoch_val_loss += val_loss.item()
            epoch_val_loss = epoch_val_loss/val_batch_len

            if min_val_loss - epoch_val_loss > val_loss_decrease_thresh:
                min_val_loss = epoch_val_loss
                best_y_test_pred = y_test_pred
                best_y_test = y_test
                best_model = copy.deepcopy(model)
                best_epoch = e_i
                best_epoch_train_loss = epoch_train_loss
                print("Epoch:", e_i, "Training loss:", epoch_train_loss, "Validation loss", epoch_val_loss)
                no_improve_epoch_count = 0
                continue

            else:
                no_improve_epoch_count += 1
                print("Epoch:", e_i, "Training loss:", epoch_train_loss, "Validation loss", epoch_val_loss,
                      "**** No improvement ****")

            if (no_improve_epoch_count == no_improve_epoch_tol) or (e_i == epochs-1):
                print("**** Stopping *****", "Best Epoch:", best_epoch, "Best Training loss:", best_epoch_train_loss,
                      "Best Validation loss", min_val_loss)
                best_y_test_pred = np.where(best_y_test_pred >= 0.5, 1, 0)
                results_eval = precision_recall_fscore_support(best_y_test, best_y_test_pred)
                print(dataset_file, results_eval)

                results_dump = open("results_dump_per_residue.txt", "a")
                results_dump.write(dataset_file + "\n" + "Epoch " + str(e_i) + " **Testing** " + str(results_eval) + "\n")
                results_dump.write("Best Epoch: " + str(best_epoch) + " Best Training loss: " + str(
                    best_epoch_train_loss) + " Best Validation loss: " + str(min_val_loss) + "\n\n")
                results_dump.close()

                torch.save(best_model.state_dict(), "saved_models/" + modelfile)
                break