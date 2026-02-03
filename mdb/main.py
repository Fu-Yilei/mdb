#!/usr/bin/env python3

import argparse, sys, os

from parse_args import parse_args

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    if args.command == "index":
        from index import index_main
        index_main(args)
    elif args.command == "create":
        from create import create_main
        create_main(args)
    elif args.command == "merge":
        from merge import merge_main
        merge_main(args)    
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))