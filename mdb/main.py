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
    elif args.command == "pca":
        from mdb.pca import pca_main
        pca_main(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))