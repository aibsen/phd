import pandas as pd

url_template_mars = r"https://mars.lco.global/?sort_value=jd&sort_order=desc&objectId="
url_template_lasair = r"https://lasair-ztf.lsst.ac.uk/object/"
keys_mars = ["objectId","time","filter","magpsf"]
keys_lasair = ["MJD",'Filter', 'magpsf','status']
output_path = 'alerce_sn_low_prob_lcs_dirty.csv'


def scrap_sns_mars(metadata_fn, url_template=url_template_mars, keys=keys_mars, output_path=output_path):
    print(metadata_fn)
    metadata = pd.read_csv(data_dir+metadata_fn+'.csv')
    ids = list(metadata.objectId.values)
    n_ids = len(ids)
    for i, id in enumerate(ids[4244:]):
        print('Object {}/{}: {}'.format(i+4244,n_ids,id))
        try:
            url = url_template+id
            tables = pd.read_html(url) # Returns list of all tables on page
            print(tables)
            df = tables[0][keys]
            print(df)
            times = list(df.time.astype('str').values)
            times = Time(times, format="iso")
            mjds = times.mjd
            df.loc[:,'time']=mjds
            df.to_csv(data_dir+output_path, mode='a', header=not os.path.exists(output_path),index=False)
        except Exception as e:
            print('Object id {} not found in url'.format(id))
            print(e)


# fn1 = 'alerce_sn_meta_low_prob'
# scrap_sns_mars(fn1)
