
import argparse
import subprocess


# define and parse arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

bem_scripts = {}
bem_scripts['build-orders-db'] = 'interaction3.bem.scripts.build_orders_database'
bem_scripts['build-translations-db'] = 'interaction3.bem.scripts.build_translations_database'
bem_scripts['t-crosstalk'] = 'interaction3.bem.scripts.simulate_transmit_crosstalk'
bem_parser = subparsers.add_parser('bem')
bem_parser.add_argument('script_name')
bem_parser.set_defaults(lookup=bem_scripts)

mfield_scripts = {}
mfield_scripts['t-beamplot'] = 'interaction3.mfield.scripts.simulate_transmit_beamplot'
mfield_scripts['t-beamplot-folding-error'] = 'interaction3.mfield.scripts.simulate_transmit_beamplot_with_folding_error'
mfield_scripts['t-beamplot-corrected-folding-error'] = 'interaction3.mfield.scripts.simulate_transmit_beamplot_with_corrected_folding_error'
mfield_parser = subparsers.add_parser('mfield')
mfield_parser.add_argument('script_name')

array_scripts = {}
array_scripts['matrix'] = 'interaction3.arrays.matrix'
array_scripts['foldable_matrix'] = 'interaction3.arrays.foldable_matrix'
array_scripts['foldable_constant_spiral'] = 'interaction3.arrays.foldable_matrix'
array_scripts['foldable_tapered_spiral'] = 'interaction3.arrays.foldable_matrix'
array_scripts['foldable_random'] = 'interaction3.arrays.foldable_matrix'
array_scripts['foldable_vernier'] = 'interaction3.arrays.foldable_vernier'
array_parser = subparsers.add_parser('array')
array_parser.add_argument('script_name')
array_parser.set_defaults(lookup=array_scripts)


def main():
    args, unknown_args = parser.parse_known_args()
    subprocess.call(['python', '-m', args.lookup[args.script_name]] + unknown_args)
