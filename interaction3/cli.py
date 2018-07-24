## interaction3 / cli.py

import argparse

import interaction3.bem.scripts
import interaction3.mfield.scripts

bem_scripts = {}
bem_scripts['t-crosstalk'] = interaction3.bem.scripts.simulate_transmit_crosstalk
bem_scripts['r-crosstalk'] = interaction3.bem.scripts.simulate_receive_crosstalk
bem_scripts['build-orders-db'] = interaction3.bem.scripts.build_orders_database
bem_scripts['build-translations-db'] = interaction3.bem.scripts.build_translations_database

mfield_scripts = {}
mfield_scripts['t-beamplot'] = interaction3.mfield.scripts.simulate_transmit_beamplot
mfield_scripts['tr-beamplot'] = interaction3.mfield.scripts.simulate_transmit_receive_beamplot

# define master parser
parser = argparse.ArgumentParser()

# define subparsers
subparsers = parser.add_subparsers()
bem_parser = subparsers.add_parser('bem')
mfield_parser = subparsers.add_parser('mfield')
array_parser = subparsers.add_parser('array')

# define bem subparser arguments
bem_parser.add_argument('script_name')
bem_parser.set_defaults(lookup=bem_scripts)

# define mfield subparser arguments
mfield_parser.add_argument('script_name')
mfield_parser.set_defaults(lookup=mfield_scripts)


def main():

    args, unknown_args = parser.parse_known_args()
    args.lookup[args.script_name].main(unknown_args)
    # print(args)
    # print(unknown_args)

    # script_name = args.pop('script_name')
    # lookup = args.pop('lookup')


