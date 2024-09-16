Design Philosophy
=================

![](https://raw.githubusercontent.com/Cosmotheka/Cosmotheka_tutorials/master/docs/src/assets/Cosmoteka_schematic_v3.png)

Cosmotheka is a single pipeline that processes catalogue level data from a wide variety of surveys and measures their angular power spectra and covariances in a consistent manner. Cosmotheka heavily relies on `NaMaster`.

Cosmotheka is designed to allow for the largest amount of modularity possible to encourage open-source community development. Inside of each module, Cosmotheka follows an object-oriented approach. Thus, given a configuration file or dictionary, the user can instantiate a class that constains all the methods needed to create a sky map (mappers), to estimate the angular power spectrum of two fields (cl) or their covariance (cov).


| Module                | function                                                                                                                             |
| -----------           | :-----------                                                                                                                         |
| ```cls/cl.py```       | Computes the Cl's requested by the user in the configuration file from the `NaMaster` fields provided by the mappers.                |
| ```cls/cov.py```      | Computes the covariance matrix of the Cl's either from the maps themselves or using the theoretical predictions of ```theory.py```.  |
| ```cls/data.py```     | Reads the user configuration file and returns an instance of the relevant mappers.                                                           |
| ```cls/theory.py```   | Computes the a theory prediction for the Cl's computed in  `cl.py` using  `pyccl` (only available for certain observables).          |
| ```cls/to_sacc.py```  | Saves all the angular power spectra as well as their covariance matrix to a `SACC` file.                                             |
| ```mappers```         | Project the catalogs into `NaMaster` fields.                        