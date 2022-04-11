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


def test_rotate_map():
    nside = 64
    ls = np.arange(3*nside)
    cl = 1/(ls+10)
    mp = hp.synfast(cl, nside)
    r = hp.Rotator(coord=['G', 'C'])
    mp1 = r.rotate_map_alms(mp)
    mp2 = xc.mappers.rotate_map(mp, r)
    assert np.all(mp1 == mp2)


def test_rotate_ellipticities():
    nside = 128
    npix = hp.nside2npix(nside)
    ls = np.arange(3*nside)
    cl = 1/(ls+10)**2
    cl0 = np.zeros(3*nside)
    _, q, u = hp.synfast([cl0, cl, cl0, cl0], nside, new=True)
    r = hp.Rotator(coord=['G', 'C'])
    qr, ur = r.rotate_map_pixel([q, u])
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    # The catalog now has positions and ellipticities
    # in Equatorial coordinates.
    cat = {'RA': ra, 'DEC': dec, 'e1': qr, 'e2': ur}

    r = hp.Rotator(coord=['C', 'G'])

    # Rotate back to Galactic coordinates and compare with
    # input maps. Differences are expected due to different
    # interpolation types used by healpy's `rotate_map_pixel`
    # and averaging by galaxy positions, so we compare against
    # half of a standard deviation of the input maps (even though
    # maps actually agree at the ~0.25*sigma level).
    ns = 32
    n1 = xc.mappers.get_map_from_points(cat, ns, rot=r)
    q1, u1 = xc.mappers.get_map_from_points(cat, ns, rot=r,
                                            qu=[cat['e1'], cat['e2']])
    q1 /= n1
    u1 /= n1
    q = hp.ud_grade(q, nside_out=ns)
    u = hp.ud_grade(u, nside_out=ns)
    std_comp = np.sqrt(np.mean(q**2+u**2))
    assert np.all(np.fabs(q-q1) < 0.5*std_comp)
    assert np.all(np.fabs(u-u1) < 0.5*std_comp)


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
