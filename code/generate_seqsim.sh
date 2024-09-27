#!/bin/bash

query=$1
name=${query:0:-6}


blastp -query $1 -db blast_db/${name}_positive -evalue 10 -out blast_db/${name}_positive
blastp -query $1 -db blast_db/${name}_negative -evalue 10 -out blast_db/${name}_negative

python3 seqsim.py --query ${name}