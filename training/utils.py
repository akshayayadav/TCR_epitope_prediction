import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def get_train_test_split(vdj_epi_df, test_size=0.25):
    vdj_epi_train_df = pd.DataFrame()
    vdj_epi_test_df = pd.DataFrame()
    vdj_epi_df['CDR3-V-J'] = vdj_epi_df['CDR3'].map(str) + "#" + vdj_epi_df['V'].map(str) + "#" + vdj_epi_df['J'].map(str)
    for vdj_name, vdj_group_df in vdj_epi_df.groupby('CDR3-V-J'):
        pos_count = len(vdj_group_df[vdj_group_df['Class'] == 1])
        if pos_count == 1:
            pos_vdj_group_df = vdj_group_df[vdj_group_df['Class'] == 1]
            vdj_epi_train_df = pd.concat([vdj_epi_train_df, pos_vdj_group_df])

            neg_vdj_group_df = vdj_group_df[vdj_group_df['Class'] == 0]
            neg_vdj_group_df_train, neg_vdj_group_df_test = train_test_split(neg_vdj_group_df, test_size=test_size)

            vdj_epi_train_df = pd.concat([vdj_epi_train_df, neg_vdj_group_df_train])
            vdj_epi_test_df = pd.concat([vdj_epi_test_df, neg_vdj_group_df_test])

        else:
            vdj_group_train_df, vdj_group_test_df = train_test_split(vdj_group_df, test_size=test_size, stratify=vdj_group_df['Class'])
            vdj_epi_train_df = pd.concat([vdj_epi_train_df, vdj_group_train_df])
            vdj_epi_test_df = pd.concat([vdj_epi_test_df, vdj_group_test_df])


    vdj_epi_train_df = vdj_epi_train_df.reset_index(drop=True)
    vdj_epi_test_df = vdj_epi_test_df.reset_index(drop=True)

    vdj_epi_train_df = vdj_epi_train_df.drop('CDR3-V-J', axis=1)
    vdj_epi_test_df = vdj_epi_test_df.drop('CDR3-V-J', axis=1)

    return [vdj_epi_train_df, vdj_epi_test_df]

def get_simple_train_test_split(vdj_epi_df, test_size=0.25):
    vdj_epi_train_df, vdj_epi_test_df = train_test_split(vdj_epi_df, test_size=test_size, stratify=vdj_epi_df['Class'])
    vdj_epi_train_df = vdj_epi_train_df.reset_index(drop=True)
    vdj_epi_test_df = vdj_epi_test_df.reset_index(drop=True)
    return [vdj_epi_train_df, vdj_epi_test_df]



def get_positive_class_weight(dataset_df):
    neg_size = len(dataset_df[dataset_df['Class'] == 0])
    pos_size = len(dataset_df[dataset_df['Class'] == 1])

    pos_weight = neg_size/pos_size
    pos_weight = torch.Tensor([pos_weight]).float()

    return pos_weight


def predict_class_label_by_majority(pred_dict, majority_thresh=19):
    y_pred_majority = list()
    for idx in pred_dict:
        preds = pred_dict[idx]
        if preds.count(1) > majority_thresh:
            y_pred_majority.append(1)
        else:
            y_pred_majority.append(0)

    return y_pred_majority