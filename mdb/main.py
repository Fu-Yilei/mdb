#!/usr/bin/env python3

import argparse, sys, os

from mdb.parse_args import parse_args

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    if args.command == "index":
        from mdb.index import index_main
        index_main(args)
    elif args.command == "create":
        from mdb.create import create_main
        create_main(args)
    elif args.command == "merge":
        from mdb.merge import merge_main
        merge_main(args)    
    elif args.command == "append":
        from mdb.merge import append_main
        append_main(args)
    elif args.command == "pca":
        from mdb.pca import pca_main
        pca_main(args)
    elif args.command == "stats":
        from mdb.stats import stats_main
        stats_main(args)
    elif args.command == "viz":
        from mdb.viz import viz_main
        viz_main(args)
    elif args.command == "asmpca":
        from mdb.asmpca import asmpca_main
        asmpca_main(args)
    elif args.command == "query":
        from mdb.query import query_main
        query_main(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
