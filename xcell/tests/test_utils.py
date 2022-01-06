import xcell as xc
import healpy as hp
import numpy as np


def test_map_from_points():
    nside = 32
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside,
                         np.arange(npix),
                         lonlat=True)
    m = xc.mappers.get_map_from_points({'RA': ra,
                                        'DEC': dec},
                                       nside)
    assert np.all(m == 1)

    # In radians
    m = xc.mappers.get_map_from_points({'RA': np.radians(ra),
                                        'DEC': np.radians(dec)},
                                       nside, in_radians=True)
    assert np.all(m == 1)


def test_get_DIR_Nz():
    # If cat_spec and cat_photo are the same,
    # DIR should return the N(z) of the spec catalog.
    cat = {'z': np.random.randn(1000),
           'rmag': 2+np.random.rand(1000),
           'imag': 2+np.random.rand(1000)}
    z, nz, nz_jk = xc.mappers.get_DIR_Nz(cat, cat,
                                         ['rmag', 'imag'],
                                         'z', [-3, 3], 10,
                                         bands_photo=['rmag', 'imag'])

    nzz, ze = np.histogram(cat['z'], range=[-3, 3], bins=10, density=True)
    assert np.all((nzz-nz)/np.amax(nzz) < 1E-10)


def test_get_beam():
    ell = np.arange(3*32)
    beam_infos = {'default': None,
                  'Gaussian': {'type': 'Gaussian',
                               'FWHM_arcmin': 0.5}}
    beam_outputs = {'default':
                    np.ones(3*32),
                    'Gaussian':
                    np.exp(-1.907e-09*ell*(ell+1))}
    for mode in beam_infos.keys():
        beam_info = beam_infos[mode]
        beamm = beam_outputs[mode]
        beam = xc.mappers.get_beam(32, beam_info)
        assert ((beam - beamm) < 1e-05).all
