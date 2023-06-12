from prepare_data.prepare_data import VdjdbDataset

vdj_dataset = VdjdbDataset("db/vdjdb_trb_mhc1_scorethres1_hs.tsv", "db/vdjdb_trb_mhc1_scorethres1_hs_pos_neg.csv")
vdj_dataset.read_vdjdb_tsv()
vdj_dataset.build_neg_vdj_epi_data()
vdj_dataset.build_unseen_test_data("db/vdjdb_trb_mhc1_scorethres1/vdjdb_trb_mhc1_scorethres1_hs_pos_neg_test.csv")
vdj_dataset.write_dataset_parts("vdjdb_trb_mhc1_scorethres1_hs_pos_neg", "db/vdjdb_trb_mhc1_scorethres1", neg_parts_num=20)
