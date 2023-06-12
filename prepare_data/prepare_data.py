import random
import re


class VdjdbDataset:
    def __init__(self, in_file, dataset_outfile):
        self.vdj_epi_neg_dict = None
        self.epi_dict = None
        self.vdj_epi_pos_dict = None
        self.vdj_dataset_tsv = in_file
        self.dataset_outfile = dataset_outfile
        self.test_data_size = 100

    def read_vdjdb_tsv(self):
        vdj_epi_pos_dict = {}
        epi_dict = {}

        with open(self.vdj_dataset_tsv, "r") as tsv_file:
            tsv_file_arr = tsv_file.readlines()
            tsv_file_arr.pop(0)
            for line in tsv_file_arr:
                line = line.rstrip()
                linearr = re.split('\t', line)
                vdj_key = linearr[2] + "#" + linearr[3] + "#" + linearr[4]
                if vdj_key not in vdj_epi_pos_dict:
                    vdj_epi_pos_dict[vdj_key] = {}
                vdj_epi_pos_dict[vdj_key][linearr[9]] = 1
                epi_dict[linearr[9]] = 1

        self.vdj_epi_pos_dict = vdj_epi_pos_dict
        self.epi_dict = epi_dict

    def build_neg_vdj_epi_data(self):
        vdj_epi_neg_dict = {}
        for vdj in self.vdj_epi_pos_dict:
            vdj_pos_epi = self.vdj_epi_pos_dict[vdj].keys()
            vdj_neg_epi_arr = [epi for epi in self.epi_dict.keys() if epi not in vdj_pos_epi]
            vdj_epi_neg_dict[vdj] = {}
            for vdj_neg_epi in vdj_neg_epi_arr:
                vdj_epi_neg_dict[vdj][vdj_neg_epi] = 1

        self.vdj_epi_neg_dict = vdj_epi_neg_dict

    def build_unseen_test_data(self, test_out_file):
        self.vdj_epi_pos_dict, pos_test_data = self.get_test_data(self.vdj_epi_pos_dict, "1")
        self.vdj_epi_neg_dict, neg_test_data = self.get_test_data(self.vdj_epi_neg_dict, "0")
        pos_test_data = "\n".join(pos_test_data)
        neg_test_data = "\n".join(neg_test_data)
        test_out = open(test_out_file,"w")

        test_out.write(pos_test_data+"\n")
        test_out.write(neg_test_data+"\n")

        test_out.close()



    @staticmethod
    def get_test_data(vdj_epi_dict, classlabel):
        test_data_arr = list()
        test_vdj_arr = random.sample(vdj_epi_dict.keys(), 100)
        for vdj_key in test_vdj_arr:
            vdj_key_out = ",".join(re.split("#", vdj_key))
            epi_arr = list(vdj_epi_dict[vdj_key].keys())
            test_data_arr.append(vdj_key_out + "," + epi_arr[0] + "," + classlabel)
            vdj_epi_dict[vdj_key].pop(epi_arr[0], None)

        return [vdj_epi_dict, test_data_arr]

    '''def write_dataset(self):
        tsv_out_file = open(self.dataset_outfile, "w")

        for vdj_key in self.vdj_epi_pos_dict:
            vdj_key_out = ",".join(re.split("#", vdj_key))
            vdj_out_pos_data = [vdj_key_out + "," + epi + ",1" for epi in list(self.vdj_epi_pos_dict[vdj_key].keys())]
            vdj_out_neg_data = [vdj_key_out + "," + epi + ",0" for epi in list(self.vdj_epi_neg_dict[vdj_key].keys())]
            vdj_out_pos_data = "\n".join(vdj_out_pos_data)
            vdj_out_neg_data = "\n".join(vdj_out_neg_data)

            vdj_out_pos_data = vdj_out_pos_data+"\n"
            vdj_out_neg_data = vdj_out_neg_data+"\n"
            tsv_out_file.write(vdj_out_pos_data)
            tsv_out_file.write(vdj_out_neg_data)
        tsv_out_file.close()'''

    def write_dataset_parts(self, outfile, dir, neg_parts_num):

        for vdj_key in self.vdj_epi_pos_dict:
            vdj_key_out = ",".join(re.split("#", vdj_key))
            vdj_out_pos_data = [vdj_key_out + "," + epi + ",1" for epi in list(self.vdj_epi_pos_dict[vdj_key].keys())]
            vdj_out_neg_data = [vdj_key_out + "," + epi + ",0" for epi in list(self.vdj_epi_neg_dict[vdj_key].keys())]

            neg_data_len = len(vdj_out_neg_data)
            neg_data_part_len = neg_data_len // neg_parts_num
            counter = 1
            for neg_i in range(0, neg_data_len, neg_data_part_len):
                out_file = open(dir+"/"+outfile+str(counter)+".csv", "a")
                part_vdj_out_pos_data = "\n".join(vdj_out_pos_data)
                part_vdj_out_neg_data = "\n".join(vdj_out_neg_data[neg_i:neg_i+neg_data_part_len])

                out_file.write(part_vdj_out_pos_data + "\n")
                out_file.write(part_vdj_out_neg_data + "\n")

                out_file.close()

                counter += 1




