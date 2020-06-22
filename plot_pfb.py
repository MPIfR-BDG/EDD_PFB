#!/usr/bin/env python
from __future__ import print_function

import pylab as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot pfb output")
parser.add_argument('filename', nargs=1)

args = parser.parse_args()

ipf = open(args.filename[0],'rb')
header = ipf.read(4096)

for line in header.split('\n'):
    if 'fftSize' in line:
        fftsize = int(line.split()[1].strip().rstrip('\x00'))
        print("Found fft size: {}".format(fftsize))
        break

data = np.fromfile(ipf, dtype=np.complex64)

nchannels = fftsize / 2
nspectra = data.size / nchannels 
print("{} spectra in data".format(nspectra))
I = data.reshape(nspectra, nchannels)

plt.figure()
plt.subplot(121)
plt.imshow(10 * np.log10(abs(I*I)), aspect='auto')
plt.xlabel('Channel No.')
plt.ylabel('Time [a.u.]')

plt.subplot(122)
plt.plot(10 * np.log10(abs(I*I).sum(axis=0)))
plt.xlabel('Channel No.')
plt.ylabel('PSD [dB]')
plt.suptitle(args.filename[0])
plt.tight_layout()
plt.show()

