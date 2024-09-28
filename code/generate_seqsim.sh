#!/bin/bash

query=$1
name=${query:0:-6}


blastp -query $1 -db blast_db/positive -evalue 10 -out blast_db/positive
blastp -query $1 -db blast_db/negative -evalue 10 -out blast_db/negative

python3 seqsim.py --query ${name}
