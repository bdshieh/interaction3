
import argparse

# define and parse arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

bem_parser = subparsers.add_parser('bem')
mfield_parser = subparsers.add_parser('mfield')
array_parser = subparsers.add_parser('array')

bem_parser.add_argument('script_name')
bem_parser.add_argument('file')
bem_parser.add_argument('spec')
bem_parser.add_argument('--threads', type=int)
bem_parser.add_argument('freqs')

mfield_parser.add_argument('script_name')
mfield_parser.add_argument('file')
mfield_parser.add_argument('spec', nargs='+')
mfield_parser.add_argument('--threads', type=int)