# mdb
A toolkit to create DNA methylation database for cross-sample comparison 


## Requirement
**Output format taken from:**
- Modkit >= v0.6
- Pb-CpG-tools >= v3.0.0

**Recommended DNA methylation pileup parameter:**
- Use `--combine-strand` in modkit pileup. 
- Use `-m` function to aggregrate 5mC and 5hmC for ONT's pileup if merging ONT with PacBio.

**mdb features**
- mdb creates a genome-wide CpG matrix to support massive computing with population-level DNA methylation signals
- Use `mdb index` to index reference genome
- Use `mdb create` to create `.mdb` files (a set of `.npy` files) for each methylBED file
- Use `mdb merge` to merge per sample `.mdb` database to form a CpGxSample matrix
- Use `mdb pca` to parform PCA on merged matrix 


        usage: mdb [-h] [-v] {index,create,merge,pca} ...

        DNA methylation database builder for quick population-level analysis.

        positional arguments:
        {index,create,merge,pca}
            index               Index all CpG locations on the reference genome
            create              Create single sample-level methylation database
            merge               mdb databases from multiple samples into a single database: COMBINE STRAND and HAPLOTYPE
            pca                 Perform PCA on the merged mdb database

        options:
        -h, --help            show this help message and exit
        -v, --version         show program's version number and exit

        Version v0.0.2