# DEScls
Repository with the Cls of galaxy clustering and weak lensing for the public DESY1 dataset.

# Usage
In order to run the code use:
`python3 pipeline/main.py pipeline/data.yml cls`
You can choose between `cls`, `cov` or `to_sacc` to compute the cls or covs of the defined tracers in `pipeline/data.yml` or to create a sacc file with the already computed cls/covs.

Note that `pipeline/cl.py`, `pipeline/cov.py` and `pipeline/to_sacc.py` can be used as scripts. For instance, you can use `python3 pipeline/cl.py pipeline/data.yml DESgc0 DESgc0` to produe the auto Cl of DESgc0.

--- 
More info about the sacc files in https://github.com/LSSTDESC/sacc
