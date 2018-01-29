# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:22:23 2017

@author: bshieh3
"""

from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog
import sys
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d
import h5py

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[0]
orange = colors[1]
red = colors[2]


def open_file_dialog():
    
    app = QApplication(sys.argv)
    widget = QWidget()
    filename, _ = QFileDialog.getOpenFileName(widget, None, None, 'All Files (*)')
    return filename


def open_files_dialog():
    
    app = QApplication(sys.argv)
    widget = QWidget()
    filenames, _ = QFileDialog.getOpenFileNames(widget, None, None, 'All Files (*)')
    return filenames


def save_file_dialog():
    
    app = QApplication(sys.argv)
    widget = QWidget()
    filenames, _ = QFileDialog.getSaveFileName(widget, None, None, 'All Files (*)')
    return filenames


def read_h5_data(filepath, save_key, add_zero_freq=False, minimal=False):

    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    with h5py.File(filepath, 'r') as root:

        freqs = np.sort(np.array([float(x) for x in root[save_key].keys()
                                  if is_float(x)]))
        nfreqs = freqs.shape[0]

        nodes = root[save_key]['nodes'][:]
        node_area = root[save_key]['nodes'].attrs['node_area']

        nnodes = root[save_key][str(freqs[0])]['x'].shape[0]
        nchannels = root[save_key][str(freqs[0])]['x_ch'].shape[0]
        nmems = root[save_key][str(freqs[0])]['x_mem'].shape[0]

    x = np.zeros((nnodes, nfreqs), dtype=np.complex128)
    
    if not minimal:
        
        x_ch = np.zeros((nchannels, nfreqs), dtype=np.complex128)
        x_mem = np.zeros((nmems, nfreqs), dtype=np.complex128)

        solve_time = np.zeros(nfreqs)
        setup_time = np.zeros(nfreqs)
        niter = np.zeros(nfreqs)
        ram_usage = np.zeros(nfreqs)

    with h5py.File(filepath, 'r') as root:

        for idx, f in enumerate(freqs):

            x_key = save_key + '/' + str(f) + '/' + 'x'
            x[:, idx] = root[x_key][:]
            
            if not minimal:
                
                x_ch_key = save_key + '/' + str(f) + '/' + 'x_ch'
                x_mem_key = save_key + '/' + str(f) + '/' + 'x_mem'

                x_ch[:, idx] = root[x_ch_key][:]
                x_mem[:, idx] = root[x_mem_key][:]

                if 'solve_time' in root[x_key].attrs.keys():
                    solve_time[idx] = root[x_key].attrs['solve_time']
                if 'setup_time' in root[x_key].attrs.keys():
                    setup_time[idx] = root[x_key].attrs['setup_time']
                if 'niter' in root[x_key].attrs.keys():
                    niter[idx] = root[x_key].attrs['niter']
                if 'ram_usage' in root[x_key].attrs.keys():
                    ram_usage[idx] = root[x_key].attrs['ram_usage']

    nnodes_per_mem = int(nnodes / nmems)

    if add_zero_freq: # Insert DC component

        freqs = np.insert(freqs, 0, values=0)
        x = np.insert(x, 0, values=0, axis=-1)
        
        if not minimal:
            x_ch = np.insert(x_ch, 0, values=0, axis=-1)
            x_mem = np.insert(x_mem, 0, values=0, axis=-1)

    nfreqs = freqs.shape[0]
    
    if not minimal:
        x_seq = x.reshape((nnodes_per_mem, nmems, nfreqs), order='F')
        nodes_seq = nodes.reshape((nnodes_per_mem, nmems, -1), order='F')

    res = {}
    res['freqs'] = freqs
    res['nodes'] = nodes
    res['node_area'] = node_area
    res['x'] = x
    
    if not minimal:
        
        res['nodes_seq'] = nodes_seq
        res['x_ch'] = x_ch
        res['x_mem'] = x_mem
        res['x_seq'] = x_seq
        res['solve_time'] = solve_time
        res['setup_time'] = setup_time
        res['niter'] = niter
        res['ram_usage'] = ram_usage

    return res


def interpolate_surfaces(meminfo, freqs, f, grid=(20,20)):

    i = np.argmin(np.abs(f - freqs))
    nptx, npty = grid
    surfs = []

    for mem in meminfo:

        x, y, _ = mem['nodes'].T
        x_length = mem['x_length']
        y_length = mem['y_length']

        z = mem['x'][:, i]

        xc = (x.min() + x.max()) / 2
        yc = (y.min() + y.max()) / 2

        xi = np.linspace(xc - x_length / 2, xc + x_length / 2, nptx)
        yi = np.linspace(yc - y_length / 2, yc + y_length / 2, npty)
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear', fill_value=0)

        #         zi = np.abs(zi)*np.cos(np.angle(zi))
        zi = np.real(zi)
        xig, yig = np.meshgrid(xi, yi)

        surfs.append((xig, yig, zi))

    return surfs


def pressure_acceleration_plot(p, acc, freqs, f, axs=None, lines=None, update=False):

    i = np.argmin(np.abs(f - freqs))
    ax, twinax = axs
    
    yleft = 20*np.log10(np.abs(p))
    yright = 20*np.log10(acc)
        
    if not update:
        
        yleft_max = np.ceil(np.max(yleft)*10.)/10. # round up to the tenth place
        yright_max = np.ceil(np.max(yright)*10.)/10. # round up to the tenth place

        p_line, = ax.plot(freqs/1e6, yleft, color=blue)
        a_line, = twinax.plot(freqs/1e6, yright, color=orange, ls=':')
        
        p_marker, = ax.plot(freqs[i]/1e6, yleft[i], color=blue, marker='o')
        a_marker, = twinax.plot(freqs[i]/1e6, yright[i], color=orange, marker='o')
        
        f_line = ax.axvline(x=freqs[i]/1e6, ls=':')

        lines = [p_line, a_line, p_marker, a_marker, f_line]

        ax.legend([p_line, a_line], ['Pressure', 'Acceleration'], loc='lower right')
        
        ax.set_xlim(0, freqs.max()/1e6)
        
        ax.set_ylim(yleft_max - 60, yleft_max)
        twinax.set_ylim(yright_max - 60, yright_max)
        
        ax.set_xlabel('Frequency (MHz)')
        
        ax.set_ylabel('Pressure (dB re 1 Pa)')
        twinax.set_ylabel('Mean acceleration (dB re 1 m/$s**2$)')
        
        # ax.set_title('Frequency = ' + str(round(freqs[i]/1e6, 2)) + ' MHz')
        
        ax.grid('on')
        
        # fig.canvas.draw()
        
        return lines
    
    else:

        p_line, a_line, p_marker, a_marker, f_line = lines

        p_marker.set_data(freqs[i]/1e6, yleft[i])
        a_marker.set_data(freqs[i]/1e6, yright[i])
        f_line.set_xdata(freqs[i]/1e6)

        # ax.set_title('Frequency = ' + str(round(freqs[i]/1e6, 2)) + ' MHz')

        # fig.canvas.update()
        # fig.canvas.flush_events()


def surface_plot(meminfo, freqs, f, cmap='RdBu_r', ax=None, update=False):
    
    surfs = interpolate_surfaces(meminfo, freqs, f)
    
    if not update:
        
        for xig, yig, zi in surfs:
            im = ax.pcolormesh(xig, yig, zi, cmap=cmap, shading='gouraud')

        zmax = []
        for mem in meminfo:

            z = mem['x']
            zmax.append(np.max(np.abs(np.real(z))))
        
        zmax = max(zmax)

        ax.set_axis_off()
        ax.set_aspect('equal')
        
    else:
        
        for im in [a for a in ax.get_children() if type(a) is mpl_toolkits.mplot3d.art3d.Poly3DCollection]:
            im.remove()
            
        for xig, yig, zi in surfs:
            im_new = ax.pcolormesh(xig, yig, zi, cmap=cmap, shading='gouraud')


def directivity_plot(dr, angles, freqs, f, ax=None, lines=None, update=False):
    
    i = np.argmin(np.abs(f - freqs))

    d = 20*np.log10(np.abs(dr[:,i])/np.max(np.abs(dr[:,i])))
    d[d < -60] = -60
    theta = np.deg2rad(angles)
    
    if not update:
        
        line, = ax.plot(theta, d)
        
        ax.set_rlim(-40, 0)
        ax.set_rticks([-40, -30, -20, -10, 0])
        ax.set_theta_zero_location('N')
    
        return [line,]

    else:
        
        lines[0].set_data(theta, d)


if __name__ == '__main__':
    pass