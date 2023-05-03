import pandas as pd
from Bio import SeqIO, Entrez
import os
from Bio.Blast.Applications import NcbipsiblastCommandline


class DBUtils:

    
    # generate a fasta file containing all the proteins in the database in the format:
    # {protein_id : protein_sequence}
    def generate_protein_fasta(self):
        
        # setting up the email address for Entrez
        Entrez.email = 'omaratyqy@gmail.com'

        # read the database into a pandas dataframe
        df = pd.read_csv("../dbs/HPRD/data.csv")

        # remove rows with self-interactions
        N_old = len(df)
        df = df[df['Interactor 1 RefSeq id'] != df['Interactor 2 RefSeq id']]
        N_new = len(df)

        # display number of removed rows as a percentage
        print(f"Removed {N_old - N_new} rows - ({round((N_old - N_new) / N_old * 100, 2)}%)")

        # create a list of RefSeq IDs
        refseq_ids = list(set(df['Interactor 1 RefSeq id'].tolist() + df['Interactor 2 RefSeq id'].tolist()))

        # create an empty dictionary to store the sequences
        sequences = {}

        # retrieve the protein sequences in batches of 500
        for i in range(0, len(refseq_ids), 500):
            refseq_ids_batch = refseq_ids[i:i+500]
            
            # convert the list of RefSeq IDs to a search term for Entrez
            search_term = ' OR '.join(str(refseq_id) for refseq_id in refseq_ids_batch)

            # search for the RefSeq IDs in the NCBI protein database
            handle = Entrez.esearch(db='protein', term=search_term, retmax=500)
            record = Entrez.read(handle)

            # retrieve the protein sequences for the matching RefSeq IDs
            if record['IdList']:
                protein_ids = record['IdList']
                handle = Entrez.efetch(db='protein', id=protein_ids, rettype='fasta', retmode='text')
                seq_records = list(SeqIO.parse(handle, 'fasta'))

                # store the sequences in the dictionary
                for seq_record in seq_records:
                    gene_symbol = seq_record.id.split('|')[0]
                    sequences[gene_symbol] = str(seq_record.seq)

            # print the current length of the dictionary divided by the total number of RefSeq IDs as a progress indicator
            progress = round(len(sequences) / len(refseq_ids) * 100, 2)
            print(f"Current progress: {progress}%")
        
        # print the result
        print(f"Total number of sequences: {len(sequences)}")

        # save the dictionary as a fasta file
        with open('../dbs/generated/sequences.fasta', 'w') as f:
            for key, value in sequences.items():
                f.write(f">{key}\n{value}\n")
    
    
    # generate the swissprot database in the format from the fasta file
    # using the command line tool makeblastdb
    def generate_swissprot_db(self):
        print("Generating swissprot database...")
        os.system(f'makeblastdb -in "../dbs/SwissProt/swissprot" -dbtype prot -out "../dbs/SwissProt/swissprot" -title "SwissProt" -parse_seqids')


    # generate the PSSM matrices of our DB by comparing the sequences to the swissprot database
    # using the command line tool psiblast
    def generate_pssm_matrices(self, input_file = "../dbs/generated/sequences.fasta", output_dir = "../dbs/generated/pssms/", evalue=0.001, num_iterations=3, num_threads=16):
        # Set the paths to the db
        db = "../dbs/SwissProt/swissprot"

        # Set the parameters for the psiblast command
        psiblast_cline = NcbipsiblastCommandline(cmd="psiblast", query=input_file, db=db, num_iterations=num_iterations, evalue=evalue, num_threads=num_threads, out_ascii_pssm=output_dir, save_each_pssm=True)

        # Run psiblast
        print("Running psiblast...")
        stdout, stderr = psiblast_cline()

        # Check the output for any error messages
        if stderr:
            print("Error running psiblast:", stderr)
        else:
            print("PSSM files generated successfully.")