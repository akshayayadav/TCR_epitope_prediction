import re


class FastaFile:
    def __init__(self, filename):
        self.filename = filename

    def get_sequence_dict(self):
        fasta = open(self.filename, "r")
        seq_dict={}
        for line in fasta:
            line = line.rstrip()
            if re.match(">", line):
                linearr = re.split("\|", line)
                seqid = linearr[1]
                seq_dict[seqid]=""

            else:
                seq_dict[seqid]=seq_dict[seqid]+line

        fasta.close()

        return seq_dict


