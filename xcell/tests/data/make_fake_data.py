import numpy as np
import healpy as hp


nside = 32
npix = hp.nside2npix(nside)

# Fake map
m = np.ones(npix)
hp.write_map("map.fits", m)

# Fake alm
alm = hp.map2alm(m)
hp.write_alm("alm.fits", alm)

# Noise file
np.savetxt("nl.txt",
           np.transpose([np.arange(3*nside),
                         np.zeros(3*nside),
                         np.ones(3*nside)]))
