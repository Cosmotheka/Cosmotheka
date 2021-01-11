import numpy as np
import healpy as hp
from astropy.io import fits
import wget


nside = 32
npix = hp.nside2npix(nside)

# Fake map
m = np.ones(npix)
hp.write_map("map.fits", m, overwrite=True)

# Fake alm
alm = hp.map2alm(m)
hp.write_alm("alm.fits", alm, overwrite=True)

# Noise file
np.savetxt("nl.txt",
           np.transpose([np.arange(3*nside),
                         np.zeros(3*nside),
                         np.ones(3*nside)]))

ra, dec = hp.pix2ang(nside, np.arange(npix),
                     lonlat=True)
on = np.ones(npix)
cols = fits.ColDefs([fits.Column(name='RA', format='D', array=ra),
                     fits.Column(name='ra', format='D', array=ra),
                     fits.Column(name='ALPHA_J2000', format='D', array=ra),
                     fits.Column(name='DEC', format='D', array=dec),
                     fits.Column(name='dec', format='D', array=dec),
                     fits.Column(name='DELTA_J2000', format='D', array=dec),
                     fits.Column(name='SG_FLAG', format='K', array=on),
                     fits.Column(name='e1', format='D', array=on),
                     fits.Column(name='e2', format='D', array=-on),
                     fits.Column(name='bias_corrected_e1', format='D',
                                 array=on),
                     fits.Column(name='bias_corrected_e2', format='D',
                                 array=-on),
                     fits.Column(name='psf_e1', format='D', array=on),
                     fits.Column(name='psf_e2', format='D', array=-on),
                     fits.Column(name='PSF_e1', format='D', array=on),
                     fits.Column(name='PSF_e2', format='D', array=-on),
                     fits.Column(name='zbin_mcal', format='D', array=-on),
                     fits.Column(name='weight', format='D', array=2*on),
                     fits.Column(name='WEIGHT_SYSTOT', format='D', array=2*on),
                     fits.Column(name='WEIGHT_CP', format='D', array=2*on),
                     fits.Column(name='WEIGHT_NOZ', format='D', array=2*on),
                     fits.Column(name='ZREDMAGIC', format='D', array=0.59*on),
                     fits.Column(name='Z', format='D', array=0.59*on)])
hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto("catalog.fits", overwrite=True)

wget.download("http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits")  # noqa