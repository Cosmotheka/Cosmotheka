# Cosmoteka
![](https://raw.githubusercontent.com/JaimeRZP/Cosmoteka_tutorials/master/docs/src/assets/cosmoteka_logo.png)
[![Coverage Status](https://coveralls.io/repos/github/xC-ell/xCell/badge.svg?branch=master)](https://coveralls.io/github/xC-ell/xCell?branch=master)
## Supported Mappers

| Mapper         | Description | Science  |  Cataloge |   
| -----------    | ----------- | -------- |  -------- |
| ```2MPZ```     |             |          |           |
| ```ACTtSZ BN```   |             |  [Madhavacheril et al, 2019](https://arxiv.org/abs/1911.05717)       | [cataloge](https://lambda.gsfc.nasa.gov/product/act/act_dr4_derived_maps_get.html)            |
| ```ACTtSZ D56```  |             |  [Madhavacheril et al, 2019](https://arxiv.org/abs/1911.05717)        | [cataloge](https://lambda.gsfc.nasa.gov/product/act/act_dr4_derived_maps_get.html)         |
| ```ACTCMB BN```   |             |  [Madhavacheril et al, 2019](https://arxiv.org/abs/1911.05717)        | [cataloge](https://lambda.gsfc.nasa.gov/product/act/act_dr4_derived_maps_get.html)            |
| ```ACTCMB D56```  |             |  [Madhavacheril et al, 2019](https://arxiv.org/abs/1911.05717)        | [cataloge](https://lambda.gsfc.nasa.gov/product/act/act_dr4_derived_maps_get.html)          |
| ```ACTk BN```     |             |  [Darwish et al, 2020](https://arxiv.org/abs/2004.01139)        | [cataloge](https://lambda.gsfc.nasa.gov/product/act/act_dr4_derived_maps_get.html)            |
| ```ACTk D56```    |             |  [Darwish et al, 2020](https://arxiv.org/abs/2004.01139)        | [cataloge](https://lambda.gsfc.nasa.gov/product/act/act_dr4_derived_maps_get.html)          |
| ```BOSS```     |  x          | [Alam et al, 2016](https://arxiv.org/abs/1607.03155)        | [cataloge](https://data.sdss.org/sas/dr12/boss/)                                                 |
| ```CIBLenz```  |  ~          | [Lenz et al, 2019](https://arxiv.org/abs/1905.00426)        | [cataloge](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8A1SR3)             |
| ```CatWISE```  |             |          |           |
| ```DELS```     |             |          |           |
| ```DESY1gc```  |  x          |  [Abbot et al, 2017](https://arxiv.org/abs/1708.01530)        | [cataloge](https://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/redmagic/)                |
| ```DESY1wl```  |  ~          | [Abbot et al, 2017](https://arxiv.org/abs/1708.01530)         | [cataloge](https://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/shear_catalogs/)          |
| ```DESY3gc```  |             |   [Abbot et al, 2021](https://arxiv.org/abs/2105.13549)       |  
| ```DESY3wl```  |             |   [Abbot et al, 2021](https://arxiv.org/abs/2105.13549)       |                                                                                                 |
| ```eBOSS```    |  x          | [Gil-Marin et al, 2018](https://arxiv.org/abs/1801.02689)         | [cataloge](https://data.sdss.org/sas/dr14/eboss/)                                                |
| ```HSC_DR1wl```|             |          |                                                                                                 |
| ```KiDS1000``` |  ~          | [Heymans et al, 2020](https://arxiv.org/abs/2007.15632)         | [cataloge](https://kids.strw.leidenuniv.nl/DR4/data_files/)                                      |
| ```NVSS```     |             |          |                                                                                                 |
| ```P15tSZ```   |  ~          | [Planck Collaboration, 2015](https://arxiv.org/abs/1502.05956)         | [cataloge](https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/ysz_index.html)      |
| ```P18CMBk```  |  ~          | [Planck collaboration, 2018](https://arxiv.org/abs/1807.06210)         |  Where did we get this from?                                                                    |
| ```P18CMB```  |  ~          | [Planck Collaboration, 2018](https://arxiv.org/abs/1807.06208)        | [cataloge](https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/foregrounds.html)    |
| ```ROSAT```    |             |          |           |
| ```SPT```      |             |          |           |
| ```WIxSC```    |             |          |           |



Pipeline to compute Nx2pt angular power spectra and their covariances.

# Usage
In order to run the code use `python3 run_cls.py input/kv450_1024.yml cls`.
You can see the different options with `python3 run_cls.py -h`.

You can run directly `xcell/cls/cl.py`, `cov.py`, `to_sacc.py` with `python3 -m` as `python3 -m xcell.cls.cl input/kv450_1024.yml KV450__0 KV450__0`.


--- 
More info about the sacc files in https://github.com/LSSTDESC/sacc
