from astropy.io import fits
from astropy.table import Table

import os
import pandas as pd
import numpy as np
import healpy as hp



class QuasarMapper:
    def __init__(self, path_datas , path_randoms, edges = [1.5], nside = 512): 
        #Inputs: 1) an array of paths to data fields
        #        2) an array of paths to random fields
        #        3) an array of bin inner boundaries  
        #        4) resolution, defaulted to 512
        
        #make sure all data paths exist can store them in a dic
        #Will this break if lists are asymmetrical?
        self.cat_data = []
        self.cat_random = []
        for file_data, file_random in zip(path_datas, path_randoms):
            if not os.path.isfile(file_data):
                raise ValueError(f"File {file_data} not found")
            with fits.open(file_data) as f:
                self.cat_data.append(Table.read(f).to_pandas()
            if not os.path.isfile(file_random):
                raise ValueError(f"File {file_random} not found")
            with fits.open(file_random) as f:
                self.cat_random.append(Table.read(f).to_pandas()

        #set resolution
        self.nside = nside
        
        #set l's per band, defaulted to 20
        self.bands = nmt.NmtBin.from_nside_linear(self.nside, 20)
        

        #Merge all data mappings into one single sky map 
        self.cat_data = pd.concat(self.cat_data)
        self.cat_random = pd.concat(self.cat_random)
        
        #Bin data                              
        self.edges = edges #inner bins edges
        self.binned   = self.__bin_z(self.cat_data ,  self.edges)
        self.binned_r = self.__bin_z(self.cat_random ,  self.edges)
        #arrays containing the binned data for each field
        
        #Store maps
        self.maps = {}
        for i in range(len(self.edges)+1):
            self.mask_field  =  self.get_mask(self.binned[i])
            self.mask_random =  self.get_mask(self.binned_r[i])
            self.alpha = self.get_alpha(self.nmean_field, self.nmean_random)
            self.nmean = self.get_nmean(self.mask_random, self.alpha)
            self.delta = self.get_delta_map(self.mask_field, self.mask_random, self.alpha)
                                       
            self.maps["bin_{}".format(i)+"_delta"]= self.delta
            self.maps["bin_{}".format(i)+"_nmean"]= self.nmean
            self.maps["bin_{}".format(i)+"_mask"] = self.mask_random
                   

        
   ###############
   ###############
        
    def __bin_z(self, cat, edges):
        #inputs: cat --> unbinned data
        #        edges -> upper boundaries of bins
        edges_full = [0.] + list(edges) + [1E300]  #David's better version
        cat_bin = [cat[(cat['Z']>=edges_full[i]) & (cat['Z']<edges_full[i+1])]
                   for i in range(len(edges)+1)]
        return cat_bin
                                       
     def get_mask(self, field):
        #imputs: pandas frames
        #Calculates the mean quasar count per pixel of the field
        field_ra = np.radians(field['RA'].values) #Phi
        field_dec = np.radians(field['DEC'].values) #Pi/2 - dec = theta

        field_FKP = np.array(field['WEIGHT_FKP'].values) 
        field_SYSTOT = np.array(field['WEIGHT_SYSTOT'].values) 
        field_CP = np.array(field['WEIGHT_CP'].values) 
        field_NOZ = np.array(field['WEIGHT_NOZ'].values)
        field_data = field_SYSTOT*field_CP*field_NOZ #FKP left out

        field_indices = hp.ang2pix(self.nside, np.pi/2 - field_dec, field_ra) #pixel_indecis

        field_pixel_data = np.bincount(field_indices, field_data, hp.nside2npix(self.nside)) 
                                                                            #for each pixel in the resolution,                                                                 #have been assigned the such pixel

        return field_pixel_data
    
    def get_nmean_map(self, mask, alpha ):
        return mask*alpha
    
    def get_alpha(self, nmean_field, nmean_random):
        #imputs: pandas frames
        #Calculates the mean quasar count per pixel field to random ratio
        
        alpha = sum(nmean_field)/sum(nmean_random)

        return alpha
    
    def get_delta_map(self, nmean_field, nmean_random, alpha):
        #inputs
        #Calculates the quasar density map of the 
        
        delta_map = np.zeros(hp.nside2npix(self.nside))
        goodpix = nmean_random > 0   #avoid dividing by 0
        delta_map[goodpix] = nmean_field[goodpix]/(alpha*nmean_random[goodpix]) - 1

        # The maps are: delta, mask, mean_number
        return delta_map
                                       
    
    def get_nl(self, mask, nmean, wsp):
        #Assumptions:
        #1) noise is uncorrelated such that the two sums over
        #pixels collapse into one --> True for poisson noise
        #2) Pixel area is a constant --> True for healpy

        #Input:
        #mask --> mask 
        #n --> map with mean number of galaxies per pixel 

        pixel_A = 4*np.pi/hp.nside2npix(self.nside)

        sum_items = np.zeros(hp.nside2npix(self.nside))
        goodpix = nmean > 0   #avoid dividing by 0
        sum_items[goodpix] = (mask[goodpix]**2/nmean[goodpix])

        #David's notes formula
        N_ell = pixel_A*np.mean(sum_items)
        
        nl_coupled = np.array([N_ell * np.ones(3*self.nside)])
        #nl_decouple = wsp.decouple_cell(nl_coupled) #removed bias here

        #Following Carlos code
        return nl_coupled
