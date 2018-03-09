
import argparse

# define and parse arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

bem_parser = subparsers.add_parser('bem')
mfield_parser = subparsers.add_parser('mfield')