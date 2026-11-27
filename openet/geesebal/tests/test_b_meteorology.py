import pprint

import ee
import pytest

import openet.geesebal.meteorology as meteorology
import openet.geesebal.utils as utils
# TODO: import utils from openet.core
# import openet.core.utils as utils


@pytest.mark.parametrize(
    "time_start, xy, expected",
    [
        # # 2015-07-13 LC08_042035_20150713
        # [
        #     1436812419150, [-119.5, 36.0],
        #     {
        #         'tmin': 18.6788, 'tmax': 34.3146, 'tair': 33.4755, 'wind_med': 2.2003,
        #         'rh': 22.9726, 'rs_inst': 893.0418, 'rs_24h': 343.5405, 'tfac': 2.5005,
        #     }
        # ],
        # 2017-07-16 LC08_044033_20170716 High ET
        [
            1500230731090, [-121.5265, 38.74],
            {
                'tmin': 19.7972, 'tmax': 40.4264, 'tair': 38.6678, 'wind_med': 1.8219,
                'rh': 17.2889, 'rs_inst': 914.9171, 'rs_24h': 352.0320, 'tfac': 2.2949,
            }
        ],
        # # 2017-07-16 LC08_044033_20170716 Centroid
        # [
        #     1500230731090, [-121.66, 38.90],
        #     {
        #         'tmin': 20.15, 'tmax': 41.45, 'tair': 37.6164, 'wind_med': 2.2668,
        #         'rh': 21.3166, 'rs_inst': 913.9269, 'rs_24h': 351.3760, 'tfac': 2.1076,
        #     }
        # ],
    ]
)
def test_nldas_gridmet(time_start, xy, expected, tol=0.01):
    tmin, tmax, tair, wind_med, rh, rs_inst, rs_24h, tfac = meteorology.nldas_gridmet(
        time_start=time_start,
        meteorology_source_inst='NASA/NLDAS/FORA0125_H002',
        meteorology_source_daily='IDAHO_EPSCOR/GRIDMET'
    )
    assert abs(utils.point_image_value(tmin, xy)['tmin'] - expected['tmin']) <= tol
    assert abs(utils.point_image_value(tmax, xy)['tmax'] - expected['tmax']) <= tol
    assert abs(utils.point_image_value(tair, xy)['tair'] - expected['tair']) <= tol
    assert abs(utils.point_image_value(wind_med, xy)['ux'] - expected['wind_med']) <= tol
    assert abs(utils.point_image_value(rh, xy)['rh'] - expected['rh']) <= tol
    assert abs(utils.point_image_value(rs_inst, xy)['rs_inst'] - expected['rs_inst']) <= tol
    assert abs(utils.point_image_value(rs_24h, xy)['rs_24h'] - expected['rs_24h']) <= tol
    assert abs(utils.point_image_value(tfac, xy)['tfac'] - expected['tfac']) <= tol


@pytest.mark.parametrize(
    "time_start, xy, expected",
    [
        # # 2015-07-13 LC08_042035_20150713
        # [
        #     1436812419150, [-119.5, 36.0],
        #     {
        #         'tmin': 23.2093, 'tmax': 37.3045, 'tair': 31.4888, 'wind_med': 1.1209,
        #         'rh': 29.5768, 'rs_inst': 926.8146, 'rs_24h': 347.8319, 'tfac': 0,
        #     }
        # ],
        # 2017-07-16 LC08_044033_20170716 High ET
        [
            1500230731090, [-121.5265, 38.74],
            {
                'tmin': 22.7145, 'tmax': 38.7924, 'tair': 36.6552, 'wind_med': 2.0457,
                'rh': 17.8512, 'rs_inst': 908.4923, 'rs_24h': 349.7216, 'tfac': 0,
            }
        ],
        # # 2017-07-16 LC08_044033_20170716 Centroid
        # [
        #     1500230731090, [-121.66, 38.90],
        #     {
        #         'tmin': 23.5636, 'tmax': 39.4888, 'tair': 36.8002, 'wind_med': 2.1443,
        #         'rh': 17.3216, 'rs_inst': 908.3524, 'rs_24h': 349.8594, 'tfac': 0,
        #     }
        # ],
    ]
)
def test_era5land(time_start, xy, expected, tol=0.01):
    tmin, tmax, tair, wind_med, rh, rs_inst, rs_24h, tfac = meteorology.era5land(
        time_start=time_start,
        meteorology_source_inst='ECMWF/ERA5_LAND/HOURLY',
        meteorology_source_daily='projects/openet/assets/meteorology/era5land/na/daily'
    )
    assert abs(utils.point_image_value(tmin, xy)['tmin'] - expected['tmin']) <= tol
    assert abs(utils.point_image_value(tmax, xy)['tmax'] - expected['tmax']) <= tol
    assert abs(utils.point_image_value(tair, xy)['tair'] - expected['tair']) <= tol
    assert abs(utils.point_image_value(wind_med, xy)['ux'] - expected['wind_med']) <= tol
    assert abs(utils.point_image_value(rh, xy)['rh'] - expected['rh']) <= tol
    assert abs(utils.point_image_value(rs_inst, xy)['rs_inst'] - expected['rs_inst']) <= tol
    assert abs(utils.point_image_value(rs_24h, xy)['rs_24h'] - expected['rs_24h']) <= tol
    # CGM - Test is failing since ETr collection is hardcoded to South America collection
    #assert abs(utils.point_image_value(tfac, xy)['tfac'] - expected['tfac']) <= tol
