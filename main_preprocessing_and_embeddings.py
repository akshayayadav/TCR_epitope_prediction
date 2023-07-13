from fasta_processing import process_fasta
from vdjdb_preprocessing import process_tsv
from embeddings import get_T5_embeddings

tr_bv = process_fasta.FastaFile("db/trbv_aa.fa")
tr_bv_seqs = tr_bv.get_sequence_dict()

tr_bj = process_fasta.FastaFile("db/trbj_aa.fa")
tr_bj_seqs = tr_bj.get_sequence_dict()

tr_b_tsv = process_tsv.VdjdbTsv('jh_data/jh_trb_mhc1_scorethres1_hs.tsv')
vdj_dict, epitope_dict = tr_b_tsv.get_vdj_and_epitope_dict(tr_bv_seqs, tr_bj_seqs)

t5_model = get_T5_embeddings.T5model()
t5_model.calculate_embeddings_vdj(vdj_dict, 100, per_seq=0)
t5_model.calculate_embeddings_epitope(epitope_dict, 100, per_seq=0)

t5_model.calculate_embeddings_vdj(vdj_dict, 100, per_seq=1)
t5_model.calculate_embeddings_epitope(epitope_dict, 100, per_seq=1)


