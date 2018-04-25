This directory conatins a test program for GASAL. First compile GASAL. To complie the test program run `make`. Running the test program without any options will print the options:
```
$./test_prog.out

Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-s] [-p] [-n] [-y] <batch1.fasta> <batch2.fasta>
Options: -a INT    match score [1]
         -b INT    mismatch penalty [4]
         -q INT    gap open penalty [6]
         -r INT    gap extension penalty [1]
         -s        also find the start position 
         -p        print the alignment results 
         -n        Number of threads 
         -y        Alignment type . Must be "local", "semi_global" or "global"  


````

`batch1.fasta` and `batch2.fasta` contain the sequences for the alignment. The sequiences in these files are aligned one-to-one, i.e. the first sequence in batch1.fasta is aligned to the first sequence in batch2.fasta, the second sequence in batch1.fasta is aligned to the second sequence in batch2.fasta, and so on. The directory also conatins sample batch1.fasta and batch2.fasta
