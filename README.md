# ITFT-Coursework-Assessment
## Files:

PRSH\_experiments.ipynb: A notebook containing code that runs market sessions, visualises results from csvs and runs statistical tests. All these experiments relate to altering k and the mutation function of base PRSH.

PRSH2\_experiments.ipynb: A notebook that runs market sessions, visualises results and runs statistical test for experiments on my altered PRSH (epsilon-greedy)

BSE.py: code for base BSE, used in experiments on base PRSH. I have written additional from 1976 to 1995 to enable easier changing of paramaters for my testing.

PRSH_improved.py: Isolated code for PRSH copied from BSE with modifications made for the extension section. This file does not run nor is it imported, it was simply designed so I could more easily make changes to PRSH. Key changes made are from line 364.

BSE_PRSH2.py: The code from PRSH_improved.py has been copied into here as an additional trader type (PRSH2). PRSH2 appears from line 1260. Changes are also made to the if/elif list of trader types that creates the traders for each session from line 2157.


