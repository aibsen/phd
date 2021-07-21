import numpy as np
import pandas as pd

class Preprocessor:
    """This class takes as input an csv file that contains light curves, reads 
    it entirely or by chunks depending on the case and can be used to call 
    functions to preprocess the light curves before they are transformed into 
    a data representation that the classification models can understand.

    Parameters
    ----------
    input_file: .csv file that contains lightcurves in the following format:
        id,mjd,flux,flux_err, passband
    chunk_size: number of lightcurves to be read. default is None, which means
        all light curves are loaded at once.
    n_passbands: int, optional. Number of passbands in an object. default is 2.
    -------
    """
    def __init__(self, input_file, chunk_size=None):
        self.input_file = input_file
        if chunk_size:
            self.data = pd.read_csv(input_file,memory_map=True, chunksize=chunk_size)
        else:
            self.data = pd.read_csv(input_file,memory_map=True)

    def drop_out_per_lc(self, percent=0.3):
        id_list = self.data.id.unique()
        for i in id_list: 
            query = 'id == {}'.format(i)
            self.data = self.data.drop(self.data.query(query).sample(frac=1-percent).index)

    def add_noise(self):
        flux_err = self.data.flux_err
        random_percent = np.random.rand(flux_err.size)
        self.data.flux_err = flux_err*random_percent

    def normalize_fluxes(self):
        flux = self.data.flux
        flux = (flux - flux.min())/(flux.max()-flux.min())
        self.data.flux = flux

    