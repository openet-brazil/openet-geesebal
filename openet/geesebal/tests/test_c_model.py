import pprint

import ee
import pytest

import openet.geesebal
import openet.geesebal.model as model
import openet.geesebal.utils as utils
# TODO: import utils from openet.core
# import openet.core.utils as utils


@pytest.mark.parametrize(
    'elev, time_start, longitude, latitude, expected',
    [
        # Note, this is not testing the slope and aspect dependency of function
        [3, 1500230731090, -121.5265, 38.74, 0.88257],
        [1000, 1500230731090, -121.5265, 38.74, 0.88257],
    ]
)
def test_cos_terrain(elev, time_start, longitude, latitude, expected, tol=0.0001):
    output = utils.constant_image_value(model.cos_terrain(
        elev=ee.Image.constant(elev),
        time_start=ee.Number(time_start),
        coords=ee.Image.constant([longitude, latitude]).rename(['longitude', 'latitude']),
    ))
    assert abs(output['cos_zn'] - expected) <= tol


@pytest.mark.parametrize(
    'tmin, tmax, elevation, expected',
    [
        [22.7145, 38.7924, 3, 30.7470],
        [20, 40, 0, 30.013],
        [20, 40, 1000, 23.513],
        [20, 40, 2000, 17.013],
    ]
)
def test_tair_dem_correction(tmin, tmax, elevation, expected, tol=0.0001):
    output = utils.constant_image_value(model.tair_dem_correction(
        tmin=ee.Image.constant(tmin),
        tmax=ee.Image.constant(tmax),
        elev=ee.Image.constant(elevation),
    ))
    assert abs(output['tair_dem'] - expected) <= tol


@pytest.mark.parametrize(
    'lst, elev, tair_dem, rh, sun_elev, time, lon, lat, expected',
    [
        [305.9179, 3, 30.7470, 17.8512, 64.2794, 1436812419150, -121.5265, 38.74, 305.2573],
        [328.2546, 3, 30.7470, 17.8512, 64.2794, 1436812419150, -121.640, 38.788, 327.5443],
    ]
)
def test_lst_correction(lst, elev, tair_dem, rh, sun_elev, time, lon, lat, expected, tol=0.0001):
    output = utils.constant_image_value(model.lst_correction(
        lst=ee.Image.constant(lst),
        elev=ee.Image.constant(elev),
        tair=ee.Image.constant(tair_dem),
        rh=ee.Image.constant(rh),
        sun_elevation=ee.Number(sun_elev),
        time_start=ee.Number(time),
        coords=ee.Image.constant([lon, lat]).rename(['longitude', 'latitude']),
    ))
    assert abs(output['lst_dem'] - expected) <= tol


@pytest.mark.parametrize(
    'elev, tair, rh, cos_z0, expected',
    [
        [3, 38.6678, 17.8512, 0.88257, 0.759462],
    ]
)
def test_tao_sw(elev, tair, rh, cos_z0, expected, tol=0.0001):
    output = utils.constant_image_value(model.tao_sw(
        elev=ee.Image.constant(elev),
        tair=ee.Image.constant(tair),
        rh=ee.Image.constant(rh),
        cos_z0=ee.Number(cos_z0),
    ))
    assert abs(output['tao_sw'] - expected) <= tol


@pytest.mark.parametrize(
    'lst, emissivity, albedo, tair, rh, elev, swdown_inst, sun_elevation, cos_zn, expected',
    [
        [305.9179, 0.9800, 0.0991, 38.6678, 17.8512, 3, 914.9171, 64.2794, 0.88257, 718.3466],
        [328.2546, 0.9517, 0.1828, 38.6678, 17.8512, 3, 914.9171, 64.2794, 0.88257, 492.0033],
    ]
)
def test_radiation_inst(
        lst, emissivity, albedo, tair, rh, elev, swdown_inst, sun_elevation, cos_zn,
        expected, tol=0.0001
):
    output = utils.constant_image_value(model.radiation_inst(
        lst=ee.Image.constant(lst),
        emissivity=ee.Image.constant(emissivity),
        albedo=ee.Image.constant(albedo),
        tair=ee.Image.constant(tair),
        rh=ee.Image.constant(rh),
        elev=ee.Image.constant(elev),
        swdown_inst=ee.Image.constant(swdown_inst),
        sun_elevation=ee.Number(sun_elevation),
        cos_zn=ee.Number(cos_zn),
    ))
    assert abs(output['rn_inst'] - expected) <= tol


@pytest.mark.parametrize(
    'rn, ndvi, albedo, lst, ndwi, flag_g, expected',
    [
        [718.3466, 0.8472, 0.0991, 305.9179, -0.7780, 'default',  52.8360],
        [483.1234, 0.1336, 0.1828, 328.2546, -0.2401, 'default', 137.1345],
        [718.3466, 0.8472, 0.0991, 305.9179, -0.7780, 'g_brazil',  79.3858],
        [483.1234, 0.1336, 0.1828, 328.2546, -0.2401, 'g_brazil', 257.0073],
    ]
)
def test_soil_heat_flux(rn, ndvi, albedo, lst, ndwi, flag_g, expected, tol=0.0001):
    output = utils.constant_image_value(model.soil_heat_flux(
        rn=ee.Image.constant(rn),
        ndvi=ee.Image.constant(ndvi),
        albedo=ee.Image.constant(albedo),
        lst=ee.Image.constant(lst),
        ndwi=ee.Image.constant(ndwi),
        flag_g=flag_g,
        year=2020,
    ))
    assert abs(output['g_inst'] - expected) <= tol


@pytest.mark.parametrize(
    'time_start, tmax, tmin, elev, rso_24h, expected',
    [
        [1500230731090, 38.7924, 22.7145, 3, 352.0320, 205.5336],
    ]
)
def test_radiation_24h(time_start, tmax, tmin, elev, rso_24h, expected, tol=0.0001):
    output = utils.constant_image_value(model.radiation_24h(
        time_start=ee.Number(time_start),
        tmax=ee.Image.constant(tmax),
        tmin=ee.Image.constant(tmin),
        elev=ee.Image.constant(elev),
        rso_24h=ee.Image.constant(rso_24h),
    ))
    assert abs(output['rad_24h'] - expected) <= tol


@pytest.mark.parametrize(
    'image_id, xy, expected',
    [
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.5265, 38.746], 1],
        # Test point on edge of field
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.52475, 38.75057], None],
    ]
)
def test_homogeneous_mask(image_id, xy, expected):
    image = openet.geesebal.Image.from_landsat_c2_sr(image_id)
    output = utils.point_image_value(model.homogeneous_mask(image.ndvi, image.proj), xy)
    if expected is not None:
        assert output['ndvi_mask'] == expected
    else:
        assert output['ndvi_mask'] is None


@pytest.mark.parametrize(
    'xy, expected',
    [
        [[-121.5265, 38.746], 1],
        # Test point on steep hillside
        [[-121.84152, 39.20989], None],
    ]
)
def test_slope_mask(xy, expected):
    output = utils.point_image_value(model.slope_mask(
        elev=ee.Image('USGS/SRTMGL1_003').select([0], ['elevation']),
        proj=ee.Projection('EPSG:32610', [30, 0, 501585, 0, -30, 4423515]),
    ), xy)
    if expected is not None:
        assert output['slp_mask'] == expected
    else:
        assert output['slp_mask'] is None


# TODO: Add parameter to cold_pixels function to control number of sample points
#   to speed up testing
@pytest.mark.parametrize(
    'image_id, ndvi, temp',
    [
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', 0.9226, 300.6153],
    ]
)
def test_cold_pixel(image_id, ndvi, temp):
    image = openet.geesebal.Image.from_landsat_c2_sr(image_id)
    xy = image.geometry.centroid(maxError=1).getInfo()
    output = utils.getinfo(model.cold_pixel(
        ndvi=image.ndvi,
        ndwi=image.ndwi,
        lst_dem=image.lst,
        elev=image.elevation,
        coords=image.coords,
        geometry_image=image.geometry,
        proj=image.proj,
        ndvi_cold=5,
        lst_cold=20,
    ))
    # pprint.pprint(output)

    # Check that the values seem reasonable
    assert abs(output['ndvi'] - ndvi) <= 0.0001
    assert abs(output['temp'] - temp) <= 0.0001
    assert output['sum'] > 1
    assert abs(output['x'] - xy['coordinates'][0]) <= 2
    assert abs(output['y'] - xy['coordinates'][1]) <= 2


@pytest.mark.parametrize(
    'image_id, tair, rh, swdown_inst, tfac',
    [
        [
            'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
            38.6678, 17.2889, 914.9171, 2.2949
        ],
    ]
)
def test_hot_pixel(image_id, tair, rh, swdown_inst, tfac):
    image = openet.geesebal.Image.from_landsat_c2_sr(image_id)
    xy = image.geometry.centroid(maxError=1).getInfo()
    rn = model.radiation_inst(
        lst=image.lst, emissivity=image.emissivity, albedo=image.albedo,
        tair=ee.Image.constant(tair), rh=ee.Image.constant(rh),
        elev=image.elevation, swdown_inst=ee.Image.constant(swdown_inst),
        sun_elevation=image.sun_elevation,
        cos_zn=model.cos_terrain(
            elev=image.elevation, time_start=image._time_start, coords=image.coords
        )
    )
    g = model.soil_heat_flux(
        rn=rn, ndvi=image.ndvi, albedo=image.albedo, lst=image.lst, ndwi=image.ndwi,
        flag_g='default', year=2017
    )
    output = utils.getinfo(model.hot_pixel(
        ndvi=image.ndvi, ndwi=image.ndwi, lst_dem=image.lst, rn=rn, g=g,
        tfac=ee.Image.constant(tfac).rename(['tfac']), elev=image.elevation,
        coords=image.coords, geometry_image=image.geometry, proj=image.proj,
        ndvi_hot=10, lst_hot=20,
    ))
    pprint.pprint(output)

    # Check that the values seem reasonable
    assert abs(output['ndvi'] - 0.2) <= 0.2
    assert output['sum'] >= 1
    assert abs(output['temp'] - 330) <= 10
    assert abs(output['x'] - xy['coordinates'][0]) <= 2
    assert abs(output['y'] - xy['coordinates'][1]) <= 2
    assert output['list_x']
    assert output['list_y']
    assert abs(output['Rn'] - 500) <= 200
    assert abs(output['G'] - 150) <= 100
    assert abs(output['tfac'] - 2.2) <= 0.5


@pytest.mark.parametrize(
    'lst_dem, savi, ux, elev, n_Ts_cold, hot_pixels, expected',
    [
        [
            305.2573, 0.4631, 1.8219, 3, 300.6153,
            {
                'temp': 334.9965, 'tfac': 2.2949, 'G': 152.0232, 'Rn': 490.5775,
                'list_x': [-121.64], 'list_y': [38.788],
            },
            55.4901
        ],
        [
            327.5443, 0.0898, 1.8219, 3, 300.6153,
            {
                'temp': 334.9965, 'tfac': 2.2949, 'G': 152.0232, 'Rn': 490.5775,
                'list_x': [-121.64], 'list_y': [38.788],
            },
            305.3389
        ],
    ]
)
def test_sensible_heat_flux(lst_dem, savi, ux, elev, n_Ts_cold, hot_pixels, expected, tol=0.0001):
    output = utils.constant_image_value(model.sensible_heat_flux(
        lst_dem=ee.Image.constant(lst_dem),
        savi=ee.Image.constant(savi),
        ux=ee.Image.constant(ux),
        n_Ts_cold=ee.Number(n_Ts_cold),
        d_hot_pixel=ee.Dictionary(hot_pixels),
        elev=ee.Image.constant(elev),
    ))
    assert abs(output['H'] - expected) <= tol


@pytest.mark.parametrize(
    'h_inst, g_inst, rn_inst, lst_dem, rn_24h, expected',
    [
        # flag_g=='default'
        [   55.4901,  52.8360, 718.3466, 305.3392, 205.5336, 6.7123],
        [  305.3389, 137.1345, 492.0033, 327.5443, 205.5336, 1.0447],
        # flag_g=='g_brazil'
        [   55.4901,  79.3858, 718.3466, 305.3392, 205.5336, 6.6870],
        [  305.3389, 257.0073, 492.0033, 327.5443, 205.5336, 0.0],
        # Check extreme values
        [-50, 50, 700, 305, 205, 7.3015],
        [  0, 50, 700, 305, 205, 7.3015],
        [650, 50, 700, 305, 205,      0],
    ]
)
def test_daily_et(h_inst, g_inst, rn_inst, lst_dem, rn_24h, expected, tol=0.0001):
    output = utils.constant_image_value(model.daily_et(
        h_inst=ee.Image.constant(h_inst),
        g_inst=ee.Image.constant(g_inst),
        rn_inst=ee.Image.constant(rn_inst),
        lst_dem=ee.Image.constant(lst_dem),
        rad_24h=ee.Image.constant(rn_24h),
    ))
    assert abs(output['et'] - expected) <= tol


# @pytest.mark.parametrize(
#     'ndvi, expected',
#     [
#         [0, 0],
#     ]
# )
# def test_et(ndvi, expected, tol=0.01):
#     output = utils.constant_image_value(model.et(
#         ndvi=ndvi,
#     ))
#     assert abs(output['et'] - expected) <= tol
