import numpy  as np

class CosmographyBase(object):
    """
    Base class for cosmography data.
    """
    data_name = None

    def __init__(self):

    def get_redshift(self):
        """
        Returns the effective redshift of the data

        Returns:
            z (Array): redshift array
        """
        raise NotImplementedError("Do not use base class")

    def get_data(self):
        """
        Returns the data array

        Returns:
            data (Array): data array
        """
        raise NotImplementedError("Do not use base class")

    def get_cov(self):
        """
        Returns the data covariance

        Returns:
            cov (Array): covariance
        """
        raise NotImplementedError("Do not use base class")

