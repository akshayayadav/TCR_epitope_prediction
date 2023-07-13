import re

def get_gene_dict_from_fasta(fastafile):
    genename_dict = {}
    fasta = open(fastafile,"r")
    for line in fasta:
        line = line.rstrip()
        if re.match("\>", line):
            linearr = re.split("\|", line)
            genename_dict[linearr[1]] = 1

    fasta.close()
    return genename_dict


def get_epitope_db_dict(vdjdb_tsv_filename):
    vdjdb_tsv = open(vdjdb_tsv_filename, "r")
    vdjdb_tsv_dict = {}
    line_count = 0
    for line in vdjdb_tsv:
        line = line.rstrip()
        if line_count>0:
            linearr = re.split("\t",line)
            vdjdb_tsv_dict[linearr[9]] = linearr[10] + "," + linearr[11]
        line_count+=1
    vdjdb_tsv.close()
    return vdjdb_tsv_dict


def read_jh_tcr(tcr_def_filename):
    tcr_def = open(tcr_def_filename,"r")
    tcr_def_list = [line.rstrip() for line in tcr_def]
    return tcr_def_list


def prepare_tcr_data(tcr_def_list, v_dict, j_dict):
    tcr_data_list = list()
    for tcr in tcr_def_list:
        tcr_list = re.split("\,", tcr)
        cdr3 = tcr_list[0]
        v_gene = tcr_list[1]
        j_gene = tcr_list[2]
        v_alleles = list()
        j_alleles = list()

        for v_allele in v_dict:
            v_allele_list = re.split("\*", v_allele)
            if v_allele_list[0] == v_gene:
                v_alleles.append(v_allele)

        for j_allele in j_dict:
            j_allele_list = re.split("\*", j_allele)
            if j_allele_list[0] == j_gene:
                j_alleles.append(j_allele)


        for v in v_alleles:
            for j in j_alleles:
                tcr_data_list.append(cdr3+","+v+","+j)

    return tcr_data_list


def pair_tcr_data_with_epitope_db(tcr_data_list, epitope_dict):
    tcr_epitope_data_list = list()
    for tcr in tcr_data_list:
        for epitope in epitope_dict:
            tcr_epitope_data_list.append(tcr+","+epitope+","+str(99))

    return tcr_epitope_data_list


def write_tcr_epitope_data_list(tcr_epitope_data_list, outfilename, type="csv"):
    outfile = open(outfilename, "w")
    if type == "csv":
        for tcr_epitope_data in tcr_epitope_data_list:
            outfile.write(tcr_epitope_data+"\n")
    elif type == "tsv":
        outfile.write("CDR3"+"\t"+"V"+"\t"+"J"+"\t"+"Epitope"+"\n")
        for tcr_epitope_data in tcr_epitope_data_list:
            tcr_epitope_data_arr = re.split("\,", tcr_epitope_data)
            outfile.write(tcr_epitope_data_arr[0]+"\t"+tcr_epitope_data_arr[1]+"\t"+tcr_epitope_data_arr[2]+"\t"+tcr_epitope_data_arr[3]+"\n")
    outfile.close()
