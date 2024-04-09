import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table
import wget
import os


nside = 32
npix = hp.nside2npix(nside)

# Fake alm
m = np.ones(npix)
alm = hp.map2alm(m)
hp.write_alm("alm.fits", alm, overwrite=True)

# Fake masks
mask1 = np.ones(npix)
mask1[int(npix*(3/4)):] = 0
hp.write_map("mask1.fits", mask1, overwrite=True)
mask2 = np.ones(npix)
mask2[:-int(npix*(3/4))] = 0
hp.write_map("mask2.fits", mask2, overwrite=True)

# Fake hm1 map
hm1 = np.repeat(np.array([np.arange(4)])-2, npix//4,
                axis=0).flatten() + np.random.randn(npix)
hp.write_map("hm1_map.fits", hm1, overwrite=True)

# Fake hm2 map
hm2 = np.repeat(np.array([np.arange(4)])-2, npix//4,
                axis=0).flatten() + np.random.randn(npix)
hp.write_map("hm2_map.fits", hm2, overwrite=True)

# Fake map
m = np.ones(npix)
hp.write_map("map.fits", [m, hm1, hm2], overwrite=True)
hp.write_map("map_auto_test.fits", [hm1, hm1, hm2], overwrite=True)

# Noise file
np.savetxt("nl.txt",
           np.transpose([np.arange(3*nside),
                         np.zeros(3*nside),
                         np.ones(3*nside)]))

ra, dec = hp.pix2ang(nside, np.arange(npix),
                     lonlat=True)
on = np.ones(npix)
half_on = np.append(np.zeros(int(npix/2)), np.ones(int(npix/2)))

ottf = np.repeat(np.array([np.arange(4)]), npix//4,
                 axis=0).flatten()

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
                                 array=ottf),
                     fits.Column(name='bias_corrected_e2', format='D',
                                 array=-ottf),
                     fits.Column(name='psf_e1', format='D', array=ottf),
                     fits.Column(name='psf_e2', format='D', array=-ottf),
                     fits.Column(name='PSF_e1', format='D', array=ottf),
                     fits.Column(name='PSF_e2', format='D', array=-ottf),
                     fits.Column(name='coadd_objects_id', format='D',
                                 array=on),
                     fits.Column(name='R11', format='D', array=on),
                     fits.Column(name='R22', format='D', array=on),
                     fits.Column(name='R12', format='D', array=on-1),
                     fits.Column(name='R21', format='D', array=on-1),
                     fits.Column(name='flags_select', format='D', array=on-1),
                     fits.Column(name='weight', format='D', array=2*on),
                     fits.Column(name='WEIGHT_SYSTOT', format='D', array=2*on),
                     fits.Column(name='WEIGHT_CP', format='D', array=2*on),
                     fits.Column(name='WEIGHT_NOZ', format='D', array=2*on),
                     fits.Column(name='ZREDMAGIC', format='D', array=0.59*on),
                     fits.Column(name='PHOTOZ_3DINFER', format='D',
                                 array=0.15*on),
                     fits.Column(name='Z', format='D', array=0.59*on),
                     fits.Column(name='Z_B', format='D', array=0.2*on),
                     fits.Column(name='Z_B_MIN', format='D', array=0*on),
                     fits.Column(name='Z_B_MAX', format='D', array=3*on),
                     fits.Column(name='GAAP_Flag_ugriZYJHKs', format='K',
                                 array=0*on)])

hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto("catalog.fits", overwrite=True)

cols = fits.ColDefs([fits.Column(name='zbin_mcal', format='D', array=on),
                     fits.Column(name='zbin_mcal_1p', format='D',
                                 array=half_on),
                     fits.Column(name='zbin_mcal_1m', format='D',
                                 array=half_on),
                     fits.Column(name='zbin_mcal_2p', format='D',
                                 array=half_on),
                     fits.Column(name='zbin_mcal_2m', format='D',
                                 array=half_on),
                     fits.Column(name='coadd_objects_id',
                                 format='D', array=[1]),
                     # Needed by test_mapper_desy1wl
                     fits.Column(name='Z_MID', format='D', array=0.6*on),
                     fits.Column(name='BIN1', format='D', array=1*on),
                     fits.Column(name='BIN2', format='D', array=2*on),
                     fits.Column(name='BIN3', format='D', array=3*on),
                     fits.Column(name='BIN4', format='D', array=4*on)
                     ])

hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto("cat_zbin.fits", overwrite=True)


with fits.open("catalog.fits") as f:
    t = Table.read(f)
    t['SG_FLAG'][:] = 0
    t.write('catalog_stars.fits', overwrite=True)

if not os.path.isfile("2pt_NG_mcal_1110.fits"):
    wget.download("http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/chains/2pt_NG_mcal_1110.fits")  # noqa


zs = np.random.randn(npix)*0.05+0.15
zs[zs < 0] = 0
mags = np.arange(npix)
cols = fits.ColDefs([fits.Column(name='SUPRA', format='D',
                                 array=ra),
                     fits.Column(name='SUPDEC', format='D',
                                 array=dec),
                     fits.Column(name='L', format='D',
                                 array=ra),
                     fits.Column(name='B', format='D',
                                 array=dec),
                     fits.Column(name='RA', format='D',
                                 array=np.radians(ra)),
                     fits.Column(name='DEC', format='D',
                                 array=np.radians(dec)),
                     fits.Column(name='ZPHOTO', format='D',
                                 array=zs),
                     fits.Column(name='ZPHOTO_CORR', format='D',
                                 array=zs),
                     fits.Column(name='ZSPEC', format='D',
                                 array=zs),
                     fits.Column(name='JCORR', format='D',
                                 array=mags),
                     fits.Column(name='KCORR', format='D',
                                 array=mags),
                     fits.Column(name='HCORR', format='D',
                                 array=mags),
                     fits.Column(name='W1MCORR', format='D',
                                 array=mags),
                     fits.Column(name='W2MCORR', format='D',
                                 array=mags),
                     fits.Column(name='BCALCORR', format='D',
                                 array=mags),
                     fits.Column(name='RCALCORR', format='D',
                                 array=mags),
                     fits.Column(name='ICALCORR', format='D',
                                 array=mags)])
hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto("catalog_2mpz.fits", overwrite=True)

tab = Table()
tab['zCorr'] = zs
tab['Zspec'] = zs
tab['ra_WISE'] = ra
tab['dec_WISE'] = dec
tab['W1c'] = mags.astype(float)
tab['W2c'] = mags.astype(float)
tab['Bcc'] = mags.astype(float)
tab['Rcc'] = mags.astype(float)
tab.write('catalog_spec_2mpz.csv', format='csv', overwrite=True)
