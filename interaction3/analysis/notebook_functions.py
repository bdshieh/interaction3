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
import sqlite3 as sql
import pandas as pd
from contextlib import closing
import os.path

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


def get_beamplot(file, angle, reshape=False):

    if _file_exists(file):
        with closing(sql.connect(file)) as con:
            query = 'SELECT x, y, z, brightness FROM image WHERE angle=? ORDER BY x, y, z'
            table = pd.read_sql(query, con, params=(angle,))
    else:
        raise IOError('File not found.')

    field_pos = np.array(table[['x', 'y', 'z']])
    brightness = np.array(table['brightness'])

    if reshape:

        x, y, z = np.atleast_2d(field_pos).T
        nx = len(np.unique(x))
        ny = len(np.unique(y))
        nz = len(np.unique(z))

        field_pos = field_pos.reshape((nx, ny, nz, -1), order='F')
        brightness = brightness.reshape((nx, ny, nz), order='F')

    return brightness, field_pos


def _file_exists(file):
    return os.path.isfile(file)


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