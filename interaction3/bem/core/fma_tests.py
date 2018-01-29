## bem / core / fma_tests.py
'''
Tester functions for fast multipole calculations. These functions are used in
scripts to determine order numbers for the translation operator and quadrature
rules. These represent the 'worst-case' translations.

Author: Bernard Shieh (bshieh@gatech.edu)
'''
import numpy as np

from . import fma_functions as fma


def nine_point_test(k, xdim, ydim, level, translation_order, theta_order, phi_order, rho, c):
    '''
    Nine point test: (1) sources are placed in a grid around a box (1 center
    source and 8 on the periphery, (2) source field is translated from the 
    source box to an evaluation box with center located two box lengths away,
    (3) translated field is evaluated at 9 points on a grid in the evaluation
    box.
    '''
    Dx, Dy = xdim / (2 ** level), ydim / (2 ** level)
    
    # set source and target box for one-box buffer scheme
    source_origin = np.array([0.0, 0.0, 0.0])
    target_origin = np.array([2 * Dx, 0.0, 0.0])
    
    # locate sources and targets at 9 points in each box
    sources = (np.c_[np.array([0.0, 0.5, 1.0, 0, 0.5, 1.0, 0.0, 0.5, 1.0]), 
        np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0 ]), 
        np.zeros(9)] * Dx - np.array([Dx / 2, Dy / 2, 0]) + source_origin)
    targets = sources + target_origin
    
    strength = np.ones(1, dtype=np.complex128)
    quadrule = fma.fft_quadrule(theta_order, phi_order)

    # calculate translation operator
    delta_r = target_origin - source_origin
    rhat = delta_r / fma.mag(delta_r)
    kcoord = quadrule['kcoord']
    weight = quadrule['weights'][0, 0]
    cos_angle = rhat.dot(kcoord.transpose((0, 2, 1)))

    translator = fma.mod_ff2nf_op(fma.mag(delta_r), cos_angle, k, translation_order)
    targets_exp_part = fma.calc_exp_part(targets, target_origin, kcoord, k)
    
    pres_exact = []
    pres_fmm = []
    
    for src in sources:

        src = np.atleast_2d(src)
        
        # FMA calculation
        src_exp_part = fma.calc_exp_part(src, source_origin, kcoord, k)
        coeffs = fma.ff_coeff(strength, src_exp_part)
        
        pres_fmm.append(weight * fma.nf_eval(coeffs * translator, targets_exp_part, k, rho, c))
        
        # direct calculation
        pres_exact.append(fma.direct_eval(strength, fma.distance(targets, src), k, rho, c))
    
    pres_exact = np.array(pres_exact).ravel()
    pres_fmm = np.array(pres_fmm).ravel()
    
    return pres_fmm, pres_exact


def calculate_error_measures(pres_fmm, pres_exact):
    '''
    Calculates error measures between exact pressure and FMA-evaluated pressure.
    '''
    # relative amplitude error in percent
    amp_rel_error = ((np.abs(pres_fmm) - np.abs(pres_exact)) / np.abs(pres_exact)) * 100
    max_amp_rel_error = np.max(np.abs(amp_rel_error))
    
    # relative phase error (re 2pi) in percent based on smallest angle 
    phase_fmm = np.angle(pres_fmm)
    phase_exact = np.angle(pres_exact)
    phase_error = np.arccos(np.round(np.cos(phase_fmm) * np.cos(phase_exact) + np.sin(phase_fmm) * np.sin(phase_exact),
                                     10))
    phase_rel_error = phase_error / (2 * np.pi) * 100
    max_phase_rel_error = np.max(np.abs(phase_rel_error))
    
    # amplitude and phase bias (useful for determing occurence of breakdown)
    amp_bias = np.mean(amp_rel_error) / np.std(amp_rel_error)
    phase_bias = np.mean(phase_rel_error) / np.std(phase_rel_error)
          
    return max_amp_rel_error, max_phase_rel_error, amp_bias, phase_bias


def print_results(max_amp_error, max_phase_error, amp_bias, phase_bias):
    
    print('Amplitude error within 0.1% ...', '[', max_amp_error <= 0.1, ']')
    print('Amplitude error within 1% ...', '[', max_amp_error <= 1, ']')
    print('Amplitude error within 5% ...', '[', max_amp_error <= 5, ']')
    print('Amplitude error within 10% ...', '[', max_amp_error <= 10, ']')
    print('Phase error within 0.1% ...', '[', max_phase_error <= 0.1, ']')
    print('Phase error within 1% ...', '[', max_phase_error <= 1, ']')
    print('Phase error within 5% ...', '[', max_phase_error <= 5, ']')
    print('Phase error within 10% ...', '[', max_phase_error <= 10, ']')
