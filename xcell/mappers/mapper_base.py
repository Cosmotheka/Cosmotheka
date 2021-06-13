import pymaster as nmt


class MapperBase(object):
    def __init__(self, config):
        self._get_defaults(config)

    def _get_defaults(self, config):
        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.nside = config['nside']
        self.nmt_field = None

    def get_signal_map(self):
        raise NotImplementedError("Do not use base class")

    def get_contaminants(self):
        return None

    def get_mask(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_coupled(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_covariance(self):
        raise NotImplementedError("Do not use base class")

    def get_nmt_field(self, **kwargs):
        if self.nmt_field is None:
            signal = self.get_signal_map(**kwargs)
            mask = self.get_mask(**kwargs)
            cont = self.get_contaminants(**kwargs)
            n_iter = kwargs.get('n_iter', 0)
            self.nmt_field = nmt.NmtField(mask, signal,
                                          templates=cont, n_iter=n_iter)
        return self.nmt_field

class MapperSDSS(MapperBase):
    
    def _get_w(self, mod='data'):
        if self.ws[mod] is None:
            cat = self.get_catalog(mod=mod)
            cat_SYSTOT = np.array(cat['WEIGHT_SYSTOT'])
            cat_CP = np.array(cat['WEIGHT_CP'])
            cat_NOZ = np.array(cat['WEIGHT_NOZ'])
            self.ws[mod] = cat_SYSTOT*cat_CP*cat_NOZ  # FKP left out
        return self.ws[mod]

    def _get_alpha(self):
        if self.alpha is None:
            w_data = self._get_w(mod='data')
            w_random = self._get_w(mod='random')
            self.alpha = np.sum(w_data)/np.sum(w_random)
        return self.alpha
    
    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = np.zeros(self.npix)
            cat_data = self.get_catalog(mod='data')
            cat_random = self.get_catalog(mod='random')
            w_data = self._get_w(mod='data')
            w_random = self._get_w(mod='random')
            alpha = self._get_alpha()
            nmap_data = get_map_from_points(cat_data, self.nside,
                                            w=w_data)
            nmap_random = get_map_from_points(cat_random, self.nside,
                                              w=w_random)
            mask = self.get_mask()
            goodpix = mask > 0
            self.delta_map = (nmap_data - alpha * nmap_random)
            self.delta_map[goodpix] /= mask[goodpix]
        return [self.delta_map]

    def get_mask(self):
        if self.mask is None:
            cat_random = self.get_catalog(mod='random')
            w_random = self._get_w(mod='random')
            alpha = self._get_alpha()
            self.mask = get_map_from_points(cat_random,
                                            self.nside_mask,
                                            w=w_random)
            self.mask *= alpha
            # Account for different pixel areas
            area_ratio = (self.nside_mask/self.nside)**2
            self.mask = area_ratio * hp.ud_grade(self.mask,
                                                 nside_out=self.nside)
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            if self.nside < 4096:
                print('calculing nl from weights')
                cat_data = self.get_catalog(mod='data')
                cat_random = self.get_catalog(mod='random')
                w_data = self._get_w(mod='data')
                w_random = self._get_w(mod='random')
                alpha = self._get_alpha()
                pixel_A = 4*np.pi/hp.nside2npix(self.nside)
                mask = self.get_mask()
                w2_data = get_map_from_points(cat_data, self.nside,
                                              w=w_data**2)
                w2_random = get_map_from_points(cat_random, self.nside,
                                                w=w_random**2)
                goodpix = mask > 0
                N_ell = (w2_data[goodpix].sum() +
                         alpha**2*w2_random[goodpix].sum())
                N_ell *= pixel_A**2/(4*np.pi)
                self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
            else:
                print('calculating nl from mean cl values')
                f = self.get_nmt_field()
                cl = nmt.compute_coupled_cell(f, f)[0]
                N_ell = np.mean(cl[2000:2*self.nside])
                self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled