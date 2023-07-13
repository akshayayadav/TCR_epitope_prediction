from jh_data.utils import get_gene_dict_from_fasta, get_epitope_db_dict, read_jh_tcr, prepare_tcr_data, \
    pair_tcr_data_with_epitope_db, write_tcr_epitope_data_list

trbv_dict = get_gene_dict_from_fasta("db/trbv_aa.fa")
trbj_dict = get_gene_dict_from_fasta("db/trbj_aa.fa")
epitope_dict = get_epitope_db_dict("db/vdjdb_trb_mhc1_scorethres1_hs.tsv")
jh_tcr = read_jh_tcr("jh_data/tcr_def.csv")

tcr_data_list = prepare_tcr_data(jh_tcr, trbv_dict, trbj_dict)
tcr_epitope_data_list = pair_tcr_data_with_epitope_db(tcr_data_list, epitope_dict)

write_tcr_epitope_data_list(tcr_epitope_data_list, "jh_data/jh_trb_mhc1_scorethres1_hs.csv", type="csv")
write_tcr_epitope_data_list(tcr_epitope_data_list, "jh_data/jh_trb_mhc1_scorethres1_hs.tsv", type="tsv")
