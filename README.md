# mdb
A toolkit to create DNA methylation database for cross-sample comparison 


## Requirement

Modkit >= v0.6
Pb-CpG-tools >= v3.0.0


        usage: mdb [-h] [-v] {index,create,merge,query} ...

        DNA methylation database builder for quick population-level analysis.

        positional arguments:
        {index,create,merge,query}
            index               Index all CpG locations on the reference genome
            create              Create population-level methylation database
            merge               mdb databases from multiple samples into a single database: COMBINE STRAND and HAPLOTYPE
            query               Query the population-level methylation database

        options:
        -h, --help            show this help message and exit
        -v, --version         show program's version number and exit

        Version v0.0.1