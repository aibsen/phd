
from re import L
import math
import pandas as pd
import numpy as np
import h5py
import george
from typing import Dict, List
from functools import partial
from astropy.table import Table, vstack
import scipy.optimize as op

# plasticc_sn_tags = {'90':0, '67':1, '52':2, '42':3, '62': 4, '95': 5}
# (r=0, g=1)
plasticc_sn_tags =[90,67,52,42,62,95]
ZTF_PB_WAVELENGTHS = {
    "1": 4804.79,
    "0": 6436.92,
}

def create_uneven_vectors(data, metadata, n_channels=2,timesteps=128, mag=False):
    #in mjd, 12 hrs = 0.5
    max_dt = 0.5

    data = data.sort_values(by=['object_id','mjd'])
    data['diff'] = data.groupby(["object_id",'passband'])["mjd"].diff().fillna(1) #the first element of each group counts as observation start
    data["gt"] = data['diff'].gt(max_dt)
    data = data.sort_values(by=['object_id', 'passband','mjd'])
    data['group'] = data['gt'].cumsum()

    if not mag: 
        data['e_log_error'] = np.exp(-np.log(data.flux_err))
        cum_err_by_group = data.groupby('group').e_log_error.sum()
        cum_err_by_group = dict(zip(cum_err_by_group.index,cum_err_by_group.values))
        data['cum_err'] = data.group.map(cum_err_by_group)
        data['weights'] = data.e_log_error/data.cum_err
        data['flux_weight'] = data.flux * data.weights

    new_flux = data.groupby('group').mean()
    metadata = metadata.sort_values(by='object_id')
    
    if not mag:
        flux_to_mag = lambda f: 30-2.5*math.log10(f)
        fluxerr_to_sigmag = lambda ferr,f: np.sqrt(np.abs(2.5/math.log(10)*(ferr/f)))

        new_flux['magpsf'] = [flux_to_mag(f) for f in new_flux.flux_weight.values]
        new_flux['sigmagpsf'] = [fluxerr_to_sigmag(ferr, f) for ferr,f in zip(new_flux.flux_weight.values, new_flux.flux_err.values)]

    #standarize 
    # new_flux['magpsf'] = (new_flux.magpsf - np.mean(new_flux.magpsf.values)/np.std(new_flux.magpsf.values))
    new_flux = new_flux.sort_values(by=['object_id','mjd'])
    
    #finally put vectors together
    # we wont take log of fluxes, since we are using mags already
    assert((metadata.object_id==new_flux.object_id.unique()).all())
    ids = new_flux.object_id.unique()
    max_length = new_flux.groupby('object_id').object_id.count().max()
    X = np.full((ids.shape[0],new_flux.passband.unique().shape[0]+2,timesteps),0)#not sure why but this is the value they used originally
    lens = np.zeros((ids.shape[0]))
    for i,id in enumerate(ids):
    
        lc = new_flux[new_flux.object_id==id]
        mag = lc.magpsf.values
        l = mag.shape[0] if mag.shape[0]<128 else 128
        mag = mag[:l]
        mjd = lc.mjd.values[:l]
        passband = lc.passband.values[:l]
        t0 = mjd.min()
        tl =mjd.max()
        l = mag.shape[0] if mag.shape[0]<128 else 128
        # print(mjd)
        normalized_mjd = ((l-1)*(mjd-t0))/(tl-t0)
        # normalized_mjd = ((timesteps-1)*(mjd-t0))/(tl-t0))
        # print(normalized_mjd)
        # print("")
        # X[i,0,0:l] = mag
        # pp = np.array([[1,0] if p==0 else [0,1] for p in passband])
        # X[i,1:3,0:l] = pp.swapaxes(1,0) 
        # X[i,3,0:l] = normalized_mjd
        # lens[i] = l
        # X[i,3,0:l] = mjd
        # l = mag.shape[0] if mag.shape[0]<128 else 128
        X[i,0,-l:] = mag
        pp = np.array([[1,0] if p==0 else [0,1] for p in passband])
        X[i,1:3,-l:] = pp.swapaxes(1,0) 
        X[i,3,-l:] = normalized_mjd
        lens[i] = l


        assert((X[i,1,-l:].astype('bool')==~X[i,2,-l:].astype(bool)).all())

    #truncate to 1st 128 observations sth like 60-128 days
    X = X[:,:,0:128] #should already have been truncated?
    # lens = lens[0:128]
    Y = metadata.true_target.values
    assert(X.shape[0]==Y.shape[0] and Y.shape[0]==ids.shape[0])
    return X, ids, Y, lens

def create_interpolated_vectors(data, length, n_channels=2):

    data_cp = data.copy()
    data_cp['ob_p']= data.object_id*10+data.passband
    # print(data_cp)

    # #sanity check, n_channels lcs per object
    print(data_cp.object_id.unique().size*n_channels)
    print(data_cp.ob_p.unique().size)
    assert(data_cp.object_id.unique().size*n_channels==data_cp.ob_p.unique().size)
    
    
    #get dataframe with min and max mjd values per each object id
    group_by_mjd = data_cp.groupby(['object_id'])['mjd'].agg(['min', 'max']).rename(columns = lambda x : 'time_' + x).reset_index()
    merged = pd.merge(data_cp, group_by_mjd, how = 'left', on = 'object_id')
    # print(merged)
    # sanity check, still same number of objects
    assert(merged.object_id.unique().size == data_cp.object_id.unique().size)
    #scale mjd according to max mjd, min mjd and the desired length of the light curve (128)
    merged['scaled_time'] = (length - 1) * (merged['mjd'] - merged['time_min'])/(merged['time_max']-merged['time_min'])
    merged['count'] = 1
    merged['cc'] = merged.groupby(['ob_p'])['count'].cumcount()
    # merged=merged.sort_values(['object_id','mjd'])
    # #sanity check
    assert(merged.object_id.unique().size==data_cp.object_id.unique().size)
    # print(merged)

    # #reshape df so that for each row there's one lightcurve (6 rows per obj) and each column is a point of it
    # # there is two main columns also, for flux and for mjd
    if 'magpsf' in merged.keys():
        unstack = merged[['object_id','ob_p', 'scaled_time', 'magpsf', 'cc']].set_index(['object_id','ob_p', 'cc']).unstack()
        units = 'magpsf'
    else:
        unstack = merged[['ob_p', 'scaled_time', 'flux', 'cc']].set_index(['ob_p', 'cc']).unstack()
        units = 'flux'

    # print(merged)
    # print(unstack)
    # print(unstack.index)
    # print(merged.object_id.unique())
    # sanity check
    assert(unstack.shape[0]== data_cp.object_id.unique().size*n_channels)
    # return
    # #transform above info into numpy arrays
    time_uns = unstack['scaled_time'].values[..., np.newaxis]
    # print(time_uns)
    flux_uns = unstack[units].values[..., np.newaxis]
    time_flux = np.concatenate((time_uns, flux_uns), axis =2)
    #create a mask to get points that are valid (not nan)
    #do this for time dim only, since fluxes will be nan when times are also
    print(time_flux.shape)
    print('just above')
    nan_masks = ~np.isnan(time_flux)[:, :, 0]
    x = np.arange(length)
    n_lcs = time_flux.shape[0]
    #here we'll store interpolated lcs
    X = np.zeros((n_lcs, x.shape[0]))
    t=range(n_lcs)
    for i in t:
        if nan_masks[i].any(): #if any point is real
            X[i] = np.interp(x, time_flux[i][:, 0][nan_masks[i]], time_flux[i][:, 1][nan_masks[i]])
        else:
            X[i] = np.zeros_like(x)
    # print(X.shape)
    n_objs = int(n_lcs/n_channels)
    #reshape vectors so the ones belonging to the same object are grouped into 6 channels
    print(X[0])
    print(X[1])
    X_per_band = X.reshape((n_objs,n_channels,length)).astype(np.float32)
    print(X_per_band[0])
    # print(X_per_band.shape)

    # print(unstack.index)
    ids = unstack.index.to_frame(index=False).object_id.unique()
    # print(ids)
    #get distance for each point to nearest real point
    X_void = np.zeros((n_lcs, x.shape[0]))
    t=range(length)
    for i in t:
        X_void[:, i] = np.abs((unstack["scaled_time"] - i)).min(axis = 1).fillna(500)

    #reshape vectors so the ones belonging to the same object are grouped into 6 channels
    X_void_per_band = X_void.reshape((n_objs,n_channels,length)).astype(np.float32)
    vectors = np.concatenate((X_per_band,X_void_per_band),axis=1)
    return vectors, ids



#modified from https://github.com/tallamjr/astronet/blob/master/astronet/preprocess.py
def predict_2d_gp(gp_predict, gp_times, gp_wavelengths):
    """Outputs the predictions of a Gaussian Process.
    Parameters
    ----------
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    gp_times : numpy.ndarray
        Times to evaluate the Gaussian Process at.
    gp_wavelengths : numpy.ndarray
        Wavelengths to evaluate the Gaussian Process at.
    Returns
    -------
    obj_gps : pandas.core.frame.DataFrame, optional
        Time, flux and flux error of the fitted Gaussian Process.
    Examples
    --------
    >>> gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    >>> number_gp = timesteps
    >>> gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
    >>> obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    >>> obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)
    ...
    """
    unique_wavelengths = np.unique(gp_wavelengths)
    number_gp = len(gp_times)
    obj_gps = []
    for wavelength in unique_wavelengths:
        gp_wavelengths = np.ones(number_gp) * wavelength
        pred_x_data = np.vstack([gp_times, gp_wavelengths]).T
        pb_pred, pb_pred_var = gp_predict(pred_x_data, return_var=True)
        # stack the GP results in a array momentarily
        obj_gp_pb_array = np.column_stack((gp_times, pb_pred, np.sqrt(pb_pred_var)))
        obj_gp_pb = Table(
            [
                obj_gp_pb_array[:, 0],
                obj_gp_pb_array[:, 1],
                obj_gp_pb_array[:, 2],
                [wavelength] * number_gp,
            ],
            names=["mjd", "flux", "flux_err", "passband"],
        )
        if len(obj_gps) == 0:  # initialize the table for 1st passband
            obj_gps = obj_gp_pb
        else:  # add more entries to the table
            obj_gps = vstack((obj_gps, obj_gp_pb))

    obj_gps = obj_gps.to_pandas()
    return obj_gps


def fit_2d_gp(
    obj_data: pd.DataFrame,
    return_kernel: bool = False,
    pb_wavelengths: Dict = ZTF_PB_WAVELENGTHS,
    **kwargs,
):
    """Fit a 2D Gaussian process.
    If required, predict the GP at evenly spaced points along a light curve.
    Parameters
    ----------
    obj_data : pd.DataFrame
        Time, flux and flux error of the data (specific filter of an object).
    return_kernel : bool, default = False
        Whether to return the used kernel.
    pb_wavelengths: dict
        Mapping of the passband wavelengths for each filter used.
    kwargs : dict
        Additional keyword arguments that are ignored at the moment. We allow
        additional keyword arguments so that the various functions that
        call this one can be called with the same arguments.
    Returns
    -------
    kernel: george.gp.GP.kernel, optional
        The kernel used to fit the GP.
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    Examples
    --------
    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}
    gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    ...
    """
    guess_length_scale = 20.0  # a parameter of the Matern32Kernel

    obj_times = obj_data.mjd.astype(float)
    obj_flux = obj_data.flux.astype(float)
    obj_flux_error = obj_data.flux_err.astype(float)
    obj_wavelengths = obj_data["passband"].astype(str).map(pb_wavelengths)

    def neg_log_like(p):  # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):  # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    # Use the highest signal-to-noise observation to estimate the scale. We
    # include an error floor so that in the case of very high
    # signal-to-noise observations we pick the maximum flux value.
    signal_to_noises = np.abs(obj_flux) / np.sqrt(
        obj_flux_error**2 + (1e-2 * np.max(obj_flux)) ** 2
    )
    scale = np.abs(obj_flux[signal_to_noises.idxmax()])

    kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel(
        [guess_length_scale**2, 6000**2], ndim=2
    )
    kernel.freeze_parameter("k2:metric:log_M_1_1")

    gp = george.GP(kernel)
    default_gp_param = gp.get_parameter_vector()
    x_data = np.vstack([obj_times, obj_wavelengths]).T
    gp.compute(x_data, obj_flux_error)

    bounds = [(0, np.log(1000**2))]
    bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
    results = op.minimize(
        neg_log_like,
        gp.get_parameter_vector(),
        jac=grad_neg_log_like,
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-6,
    )

    if results.success:
        gp.set_parameter_vector(results.x)
    else:
        # Fit failed. Print out a warning, and use the initial guesses for fit
        # parameters.
        obj = obj_data["object_id"][0]
        print("GP fit failed for {}! Using guessed GP parameters.".format(obj))
        gp.set_parameter_vector(default_gp_param)

    gp_predict = partial(gp.predict, obj_flux)

    if return_kernel:
        return kernel, gp_predict
    return gp_predict



def generate_gp_single_event(
    df: pd.DataFrame, timesteps: int = 100, pb_wavelengths: Dict = ZTF_PB_WAVELENGTHS,
    var_length = False
) -> pd.DataFrame:
    """Intermediate helper function useful for visualisation of the original data with the mean of
    the Gaussian Process interpolation as well as the uncertainity.
    Additional steps required to build full dataframe for classification found in
    `generate_gp_all_objects`, namely:
        ...
        obj_gps = pd.pivot_table(obj_gps, index="mjd", columns="filter", values="flux")
        obj_gps = obj_gps.reset_index()
        obj_gps["object_id"] = object_id
        ...
    To allow a transformation from:
        mjd	        flux	    flux_error	filter
    0	0.000000	19.109279	0.176179	1(ztfg)
    1	0.282785	19.111843	0.173419	1(ztfg)
    2	0.565571	19.114406	0.170670	1(ztfg)
    to ...
    filter	mjd	        ztfg    ztfr	object_id
    0	    0	        19.1093	19.2713	27955532126447639664866058596
    1	    0.282785	19.1118	19.2723	27955532126447639664866058596
    2	    0.565571	19.1144	19.2733	27955532126447639664866058596
    Examples
    --------
    obj_gps = generate_gp_single_event(data)
    ax = plot_event_data_with_model(data, obj_model=_obj_gps, pb_colors=ZTF_PB_COLORS)
    """

    filters = df["passband"].astype(str)
    filters = list(np.unique(filters))

    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}

    gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)

    if var_length:
        mjd_diff = max(df['mjd'])-min(df['mjd'])
        number_gp = timesteps if mjd_diff>timesteps else int(np.floor(mjd_diff))
    else:
        number_gp = timesteps
    gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    obj_gps["passband"] = obj_gps["passband"].map(inverse_pb_wavelengths)
    obj_gps["passband"] = obj_gps["passband"].astype(int)

    return obj_gps, number_gp

def create_gp_interpolated_vectors(
    object_list: List[str],
    obs_transient: pd.DataFrame,
    obs_metadata: pd.DataFrame,
    timesteps: int = 100,
    pb_wavelengths: Dict = ZTF_PB_WAVELENGTHS,
    var_length = False
) -> pd.DataFrame:
    """Generate Gaussian Process interpolation for all objects within 'object_list'. Upon
    completion, a dataframe is returned containing a value for each time step across each passband.
    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    obs_transient: pd.DataFrame
        Dataframe containing observational points with the transient section of the full light curve
    timesteps: int
        Number of points one would like to interpolate, i.e. how many points along the time axis
        should the Gaussian Process be evaluated
    pb_wavelengths: Dict
        A mapping of passbands and the associated wavelengths, specific to each survey. Current
        options are ZTF or LSST
    Returns
    -------
    df: pd.DataFrame(data=adf, columns=obj_gps.columns)
        Dataframe with the mean of the GP for N x timesteps
    Examples
    --------
    ?>>> object_list = list(np.unique(df["object_id"]))
    ?>>> obs_transient, object_list = __transient_trim(object_list, df)
    ?>>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    filters = obs_transient["passband"]
    filters = list(np.unique(filters))

    columns = []
    columns.append("mjd")
    # for filt in filters:
    #     columns.append(filt)
    columns.append("object_id")
    columns.append("passband")

    adf = pd.DataFrame(
        data=[],
        columns=columns,
    )
    id_list = []
    targets = []
    n_lcs = len(object_list)
    n_channels = 2
    X = np.ones((n_lcs, n_channels, timesteps)) #if flux is negative, set it to 1, so it can be converted to mag
    lens = np.zeros((n_lcs,))
    #if flux_err is negative, make it positive
    for i,object_id in enumerate(object_list):
        print(f"OBJECT ID:{object_id} at INDEX:{object_list.index(object_id)}")
        df = obs_transient[obs_transient["object_id"] == object_id]

        obj_gps, lc_length = generate_gp_single_event(df, timesteps, pb_wavelengths, var_length=var_length)
        # print(obj_gps)

        obj_gps = pd.pivot_table(obj_gps, index="mjd", columns="passband", values="flux")
        X[i,0,-lc_length:] = obj_gps[0]
        X[i,1,-lc_length:] = obj_gps[1]
        id_list.append(object_id)
        true_target = obs_metadata[obs_metadata.object_id==object_id].true_target.values[0]
        targets.append(true_target)
        lens[i] = lc_length
    #     obj_gps = obj_gps.reset_index()
        # obj_gps["object_id"] = object_id
        # adf = np.vstack((adf, obj_gps))
    
    X = np.where(X>0,X,1)
    # print(X)
    # print(X)
    return X, id_list, targets, lens
    # return pd.DataFrame(data=obj_gps, columns=obj_gps.columns)

def append_vectors(dataset,outputFile):
    with h5py.File(outputFile, 'a') as hf:
        X=dataset["X"]
        hf["X"].resize((hf["X"].shape[0] + X.shape[0]), axis = 0)
        hf["X"][-X.shape[0]:] = X

        ids = dataset["ids"]
        hf["ids"].resize((hf["ids"].shape[0] + ids.shape[0]), axis = 0)
        hf["ids"][-ids.shape[0]:] = ids

        Y=dataset["Y"]
        hf["Y"].resize((hf["Y"].shape[0] + Y.shape[0]), axis = 0)
        hf["Y"][-Y.shape[0]:] = Y
        hf.close()


def save_vectors(dataset, outputFile):
    hf=h5py.File(outputFile,'w')

    print("writing X")
    hf.create_dataset('X',data=dataset['X'],compression="gzip", chunks=True, maxshape=(None,None,None,))

    print("writing ids")
    hf.create_dataset('ids',data=dataset['ids'],dtype='int64',compression="gzip", chunks=True, maxshape=(None,))
    
    print("writing Y")
    hf.create_dataset('Y',data=dataset['Y'],compression="gzip", chunks=True, maxshape=(None,))

    if 'lens' in dataset.keys():
        print("writing lens")
        hf.create_dataset('lens',data=dataset['lens'],dtype='int64',compression="gzip", chunks=True, maxshape=(None,))
    
    hf.close()


