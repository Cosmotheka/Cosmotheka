#!/bin/bash

mkdir -p data
cd data
if [ ! -f eBOSS_QSO_clustering_data-NGC-vDR16.fits ] ; then
    wget https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/eBOSS_QSO_clustering_data-NGC-vDR16.fits
fi
if [ ! -f eBOSS_QSO_clustering_random-NGC-vDR16.fits ] ; then
    wget https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/eBOSS_QSO_clustering_random-NGC-vDR16.fits
fi
if [ ! -f eBOSS_QSO_clustering_data-SGC-vDR16.fits ] ; then
    wget https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/eBOSS_QSO_clustering_data-SGC-vDR16.fits
fi
if [ ! -f eBOSS_QSO_clustering_random-SGC-vDR16.fits ] ; then
    wget https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/eBOSS_QSO_clustering_random-SGC-vDR16.fits
fi
