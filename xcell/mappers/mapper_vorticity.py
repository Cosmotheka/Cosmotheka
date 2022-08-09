import numpy as np
import healpy as hp
import pymaster as nmt
import fitsio
import pyccl
from .utils import get_map_from_points, rotate_mask
from sklearn.neighbors import NearestNeighbors, BallTree
from .mapper_base import MapperBase

class MapperVorticity(MapperBase):
    def __init__(self, config):
        self._get_defaults(config)
        self.npix = hp.nside2npix(self.nside)
        self.rot = self._get_rotator('C')
        if self.rot is not None:
            raise NotADirectoryError('MapperVorticity only available for Celestial Coordinates')

        # Angular mask
        self.vort = None
        self.nl_coupled = None

    def get_catalog_and_coords(self):
        cats = []
        for file in self.config['data_catalogs']:
            d = fitsio.read(file, columns=['RA', 'DEC', 'Z'])
            z = d['Z']
            d = d[~np.isnan(z) * z>=0]
            cats.append(d)
        cats = np.hstack(cats)
        cosmo = pyccl.CosmologyVanillaLCDM()
        a = 1/(1+cats['Z'])
        r = a*pyccl.comoving_radial_distance(cosmo, a)
        theta = np.radians(90-cats['DEC'])
        phi = np.radians(cats['RA'])
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        coords = np.array([x, y, z]).T
        return cats, coords

    def _get_vorticity_maps_and_mask(self):
        if self.vort is None:
            # Read the catalog
            cat, coords = self.get_catalog_and_coords()
            # Find pairs
            pair_distances, pair_indices = self.get_pairs(cat, coords)
            # Compute vorticity for each pair
            vort = self.get_vorticity(cat, coords, pair_indices)
            # Angular coords of the CM for each pair
            xyz_cm = 0.5*(coords[pair_indices[:, 0]]+
                          coords[pair_indices[:, 1]])
            ra_cm, dec_cm = hp.vec2ang(xyz_cm, lonlat=True)

            # Get number of pairs per pixel
            nc_map = get_map_from_points({'RA': ra_cm, 'DEC': dec_cm},
                                         self.nside, ra_name='RA', dec_name='DEC')
            vort1 = get_map_from_points({'RA': ra_cm, 'DEC': dec_cm}, self.nside,
                                        w=vort[:, 0], ra_name='RA', dec_name='DEC')
            vort2 = get_map_from_points({'RA': ra_cm, 'DEC': dec_cm}, self.nside,
                                        w=vort[:, 1], ra_name='RA', dec_name='DEC')

            seen = nc_map > 0
            vort1[seen] /= nc_map[seen]
            #vort1[~seen] = hp.UNSEEN
            vort2[seen] /= nc_map[seen]
            #vort2[~seen] = hp.UNSEEN
            self.vort = [vort1, vort2]
            self.mask = nc_map
        return self.vort, self.mask

    def get_mask(self):
        return self._get_vorticity_maps_and_mask()[1]

    def get_pairs(self, cat, coords):
        radius = self.config.get('radius_max', 1.0)
        ball_tree = BallTree(coords)
        distances, indices = ball_tree.query(coords, 2, return_distance = True, sort_results = True)
        isolated_distances = []
        isolated_indices = []

        for dist, ind in zip(distances, indices):
            if dist[1] <= radius:
                isolated_distances.append(dist)
                isolated_indices.append(ind)

        isolated_indices = np.array(isolated_indices)

        pair_distances = []
        pair_indices = []

        for i1, i2 in isolated_indices:
            if ((i1,i2) in pair_indices) or ((i2, i1) in pair_indices):
                continue
            if indices[i2][1] == i1:
                pair_indices.append((i1,i2))
                pair_distances.append(distances[i1][1])

        pair_indices = np.array(pair_indices)
        pair_distances = np.array(pair_distances)
        return pair_distances, pair_indices

    def get_vorticity(self, cat, coords, pair_indices):
        vort = []
        for i1, i2 in pair_indices:
            phi1 = np.radians(cat['RA'][i1])
            phi2 = np.radians(cat['RA'][i2])
            theta1 = np.radians(90-cat['DEC'][i1])
            theta2 = np.radians(90-cat['DEC'][i2])
            z1 = cat['Z'][i1]
            z2 = cat['Z'][i2]
            dphi = phi1-phi2
            dtheta = theta1-theta2
            dz = z1-z2
            cm = 0.5*(coords[i1] + coords[i2])
            r_cm = np.sqrt(np.sum(cm**2))
            dr2 = np.sum((coords[i1]-coords[i2])**2)
            sinth = np.sin(hp.vec2ang(cm)[0])[0]
            rphi = dphi*r_cm*sinth
            rtheta = dtheta*r_cm
            vort.append([dz*rphi/dr2,-dz*rtheta/dr2])
        vort = np.array(vort)
        return vort

    def get_signal_map(self):
        return self._get_vorticity_maps_and_mask()[0]

    def print_signal_maps(self):
        if self.vort is None:
            self.vort = self.get_vorticity_maps_and_mask()[0]
        else:
            hp.mollview(self.vort[0], title = "Vorticity Map 1")
            hp.mollview(self.vort_map[1], "Vorticity Map 2")
            hp.graticule()

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            cat, coords = self.get_catalog_and_coords()
            pair_distances, pair_indices = self.get_pairs(cat, coords)
            xyz_cm = 0.5*(coords[pair_indices[:, 0]]+
                          coords[pair_indices[:, 1]])
            ra_cm, dec_cm = hp.vec2ang(xyz_cm, lonlat=True)


            vort = self.get_vorticity(cat, coords, pair_indices)
            w2s2 = get_map_from_points({'RA': ra_cm, 'DEC': dec_cm},
                                       self.nside,
                                       w=0.5*np.sum(vort**2, axis=1),
                                       ra_name='RA',
                                       dec_name='DEC', rot=self.rot)
            N_ell = hp.nside2pixarea(self.nside) * np.sum(w2s2) / self.npix
            nl = N_ell * np.ones(3*self.nside)
            nl[:1] = 0  # Ylm = for l < spin
            self.nl_coupled = np.array([nl, 0*nl, 0*nl, nl])
        return self.nl_coupled

    def get_spin(self):
        return 1
