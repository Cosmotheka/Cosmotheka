import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

import os
import pandas as pd
import collections
import numpy as np
import healpy as hp

import pymaster as nmt


class QuasarMapper:
    def __init__(self): 
        #Inputs: the four fits files from eBOSS DR14
        
        data_path = '/home/zcapjru/PhD/Data/'
        #Check for data files:
        if os.path.exists(data_path+'eBOSS_QSO_clustering_data-NGC-vDR16.fits') :
            print('NGC data found')
            self.NGC_data =  fits.open(data_path + 'eBOSS_QSO_clustering_data-NGC-vDR16.fits')
        else :
            print('missing NGC data')

        if os.path.exists(data_path+'eBOSS_QSO_clustering_data-SGC-vDR16.fits') :
            print('SGC data found')
            self.SGC_data =  fits.open(data_path + 'eBOSS_QSO_clustering_data-SGC-vDR16.fits')
        else :
            print('missing SGC data')
           
        if os.path.exists(data_path+'eBOSS_QSO_clustering_random-NGC-vDR16.fits') :
            print('NGC random data found')
            self.NGC_r_data =  fits.open(data_path + 'eBOSS_QSO_clustering_random-NGC-vDR16.fits')
        else :
            print('missing NGC random data')
           
        if os.path.exists(data_path+'eBOSS_QSO_clustering_random-SGC-vDR16.fits') :
            print('SGC random data found')
            self.SGC_r_data =  fits.open(data_path + 'eBOSS_QSO_clustering_random-SGC-vDR16.fits')
        else :
            print('missing SGC random data')
        

        self.nside = 512
        #set resolution
        
        self.bands = nmt.NmtBin.from_nside_linear(self.nside, 20)
        #set l's per band
        
        self.NGC = Table.read(self.NGC_data).to_pandas()
        self.NGC_r = Table.read(self.NGC_r_data).to_pandas()
        self.SGC = Table.read(self.SGC_data).to_pandas()
        self.SGC_r = Table.read(self.SGC_r_data).to_pandas()
        #from .fits to table to managable pandas frames

        #Merge NGC + SGC
        self.edges = [1.5] #inner bins edges
        self.whole =  pd.concat([self.NGC, self.SGC])
        self.whole_r =  pd.concat([self.NGC_r, self.SGC_r])
        
        self.binned   = self.bin_z(self.whole ,  self.edges)
        self.binned_r = self.bin_z(self.whole_r ,  self.edges)
        #arrays containing the binned data for each field
        
        self.maps = {}
        for i in range(len(self.edges)+1):
            self.nmean_field  =  self.get_nmean_map(self.binned[i])
            self.nmean_random =  self.get_nmean_map(self.binned_r[i])
            self.alpha = self.get_alpha(self.nmean_field, self.nmean_random)
            
            self.maps["bin_{}".format(i)+"_delta"]= self.get_delta_map(self.nmean_field, self.nmean_random, self.alpha) 
            self.maps["bin_{}".format(i)+"_nmean"]= self.nmean_random*self.alpha
            self.maps["bin_{}".format(i)+"_mask"] = self.nmean_random
               
        print('Maps computed succesfully')
        
        self.cls = {}
        for i in range(len(self.edges)+1):
            for j in range(len(self.edges)+1):
                if j>=i:
                    self.f_i = nmt.NmtField(self.maps["bin_{}".format(i)+"_mask"], [self.maps["bin_{}".format(i)+"_delta"]])
                    self.f_j = nmt.NmtField(self.maps["bin_{}".format(j)+"_mask"], [self.maps["bin_{}".format(i)+"_delta"]])
                    self.wsp = nmt.NmtWorkspace()
                    self.wsp.compute_coupling_matrix(self.f_i, self.f_j, self.bands)
                    self.cl = self.get_cl(self.f_i, self.f_j, self.wsp)
                    self.cls["cl_{}".format(i)+"{}".format(j)]= self.cl
                    
                    if i==j:
                        self.mask  = self.maps["bin_{}".format(i)+"_mask"]
                        self.nmean = self.maps["bin_{}".format(i)+"_nmean"]
                        self.nl = self.get_nl(self.mask, self.nmean, self.wsp)
                        self.cls["nl_{}".format(i)+"{}".format(j)]   = self.cl
                        self.cls["cl-nl_{}".format(i)+"{}".format(j)]= self.nl
                    else:
                        self.cls["nl_{}".format(i)+"{}".format(j)]   = np.zeros(len(self.cl))
                        self.cls["cl-nl_{}".format(i)+"{}".format(j)]= self.cl
                        
        print('Cls computed succesfully')       

        
   ###############
   ###############
        
    def bin_z(self, cat, edges):
        #inputs: cat --> unbinned data
        #        edges -> upper boundaries of bins
        edges_full = [0.] + list(edges) + [1E300]  #David's better version
        cat_bin = [cat[(cat['Z']>=edges_full[i]) & (cat['Z']<edges_full[i+1])]
                   for i in range(len(edges)+1)]
        return cat_bin
    
    def get_nmean_map(self, field):
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
    
    def get_cl(self, f_a, f_b, wsp):
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        # Decouple power spectrum into bandpowers inverting the coupling matrix
        cl_decoupled = wsp.decouple_cell(cl_coupled) #removed bias here

        return cl_decoupled
    
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
        nl_decouple = wsp.decouple_cell(nl_coupled) #removed bias here

        #Following Carlos code
        return nl_decouple

output = QuasarMapper()

f = open("cls.txt","w")
f.write( str(output.cls) )
f.close()

f = open("maps.txt","w")
f.write( str(output.maps) )
f.close()
