
import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import os
import subprocess
import shutil
import sys
sys.path.append('/gpfs/commons/home/mgarbulowski/homic_package/src') # to load our package
from homic import file_readers, kraken2, process_data

# db creation for mouse
# db_path = "/proj/berzelius-2024-407/users/x_magar/dbs/mm_wgs"
# ref_path = "/proj/berzelius-2024-#407/users/x_magar/references/GCF_000001635.27_GRCm39_genomic.fna"
# kraken2.prepare_db(db_path, ref_path)


# call the fun
path = "/proj/berzelius-2024-407/users/x_magar/data/spf_data"

file1 = path + "/all_merged_1.fastq"
file2 = path + "/all_merged_2.fastq"

process_data.prepare_reference(path, file1, file2)

breakpoint()


# 1 detect names of files for the analysis
data_path = "/proj/berzelius-2024-407/users/x_magar/data/spf_data/processed"
lds = np.array(os.listdir(data_path))

files_r1 = np.array([("mic_1." in file) for file in lds])
files_r2 = np.array([("mic_2." in file) for file in lds])

if (not any(files_r1 + files_r2)):
    files_r1 = [("mic_R1." in file) for file in lds]
    files_r2 = [("mic_R2." in file) for file in lds]

if (not any(files_r1 + files_r2)):
    print("Files name should contain 'mic_1' or 'mic_2' at the end of each file. Alternatively 'mic_R1' or 'mic_R1'. For instance, file_1.fastq or file_R1.fastq")
    sys.exit(1)

files_name_r1 = sorted(lds[files_r1].tolist())
files_name_r2 = sorted(lds[files_r2].tolist())

paths_r1 = [data_path + "/" + file for file in files_name_r1]
paths_r1 = ','.join(paths_r1)

paths_r2 = [data_path + "/" + file for file in files_name_r2]
paths_r2 = ','.join(paths_r2)


#fasta_string = open(data_path + "/" + "out_assembly" + "/final.contigs.fa").read()
fasta_string = open(data_path + "/SRR25456944_mic_contigs.fastq").read()

#records = list(SeqIO.parse(data_path + "/SRR25456944_mic_contigs.fastq", "fasta"))
#nuc_seq = [str(nuc.seq) for nuc in records]

#print(nuc_seq[0:2])


start2 = time.time()
result_stream = Blast.qblast("blastn", "nt", fasta_string[0:400000], format_type = "XML", ncbi_gi = False, hitlist_size=1, alignments = 1, entrez_query="txid2[ORGN]", expect = 10E-6) # txid2 is for bacteria
end2 = time.time()

print("BLAST query time:")
print(end2 - start2)


#blast_record = Blast.read(result_stream)
blast_record = result_stream.read()

blast_record = blast_record.decode()


#descs = re.findall(r"<Hit>(.*?)</Hit>", blast_record, re.DOTALL)
descs = re.findall(r"<Hit_def>(.*?)</Hit_def>", blast_record, re.DOTALL)
evs = re.findall(r"<Hsp_evalue>(.*?)</Hsp_evalue>", blast_record, re.DOTALL)

descs = list(map(lambda x: x.replace('MAG:',''),descs))
descs = list(map(lambda x: x.replace('TPA_exp:',''),descs))



taxai = [desc.split()[:4] for desc in descs]
taxai_df = pd.DataFrame(taxai, columns = ['info_I', 'info_II', 'info_III', 'info_IV']) 

print(taxai_df)

print("Genera: ")
print(set(taxai_df['info_I'].tolist()))

print("Species: ")
specs = [str(x) + " " + str(y) for x, y in zip(taxai_df['info_I'].tolist(), taxai_df['info_II'].tolist())]
print(set(specs))

fname = "/proj/berzelius-2024-407/users/x_magar/homic/files/65sp.txt"
with open(fname) as file:
    spec65 = [line.rstrip() for line in file]

res = list(set(spec65) & set(specs))
print(len(res)/len(spec65)*100)



breakpoint()



niter = 0

evals = []
seqids = []
descs = []
bss = []
nms = []
ides = []
for alignments in blast_record:
    for alignment in alignments:
            seqids.append(alignment.target.id)
            nms.append(alignment.target.name)
            descs.append(alignment.target.description)
            evals.append(alignment.annotations['evalue'])
            bss.append(alignment.annotations['bit score'])
            ides.append(alignment.annotations['identity'])
            niter = niter + 1


descs = list(map(lambda x: x.replace('MAG:',''),descs))
descs = list(map(lambda x: x.replace('TPA_exp:',''),descs))

print(type(descs))

data = {'sequence_id': seqids, 'name': nms, 'description': descs, 'e_value': evals, 'identity': ides, 'bit_score': bss}
blast_report = pd.DataFrame(data=data)

print("Found " + str(niter) + " alignments")

filtered_blast_report = blast_report[blast_report['e_value'] <= 10E-6]
print(filtered_blast_report)

descs = filtered_blast_report['description'].tolist()
print(descs)

taxai = [desc.split()[:4] for desc in descs]
taxai_df = pd.DataFrame(taxai, columns = ['info_I', 'info_II', 'info_III', 'info_IV']) 

print(taxai_df)

print("Genera: ")
print(set(taxai_df['info_I'].tolist()))

print("Species: ")
specs = [str(x) + " " + str(y) for x, y in zip(taxai_df['info_I'].tolist(), taxai_df['info_II'].tolist())]
print(set(specs))

fname = "/proj/berzelius-2024-407/users/x_magar/homic/files/65sp.txt"
with open(fname) as file:
    spec65 = [line.rstrip() for line in file]

res = list(set(spec65) & set(specs))
print(len(res)/len(spec65)*100)



# 5. binning
## MetaBat2
#metabat2 -i assembly.fasta -a depth.txt -o bins_dir/bin

#cmd = ["metabat2",
#        "-i",
#        data_path + "/" + "out_assembly" + "/final.contigs.fa",
#