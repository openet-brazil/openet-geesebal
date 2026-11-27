from datetime import datetime, timezone
import logging
import pprint

import ee
import pytest

import openet.geesebal as model
import openet.geesebal.utils as utils
# TODO: import utils from openet.core
# import openet.core.utils as utils


# TODO: Try moving to conftest and/or make a fixture
COLL_ID = 'LANDSAT/LC08/C02/T1_L2'
SCENE_ID = 'LC08_044033_20170716'
IMAGE_ID = f'{COLL_ID}/{SCENE_ID}'
SCENE_TIME = 1500230731090
SCENE_DT = datetime.fromtimestamp(SCENE_TIME / 1000.0, tz=timezone.utc)
SCENE_DATE = SCENE_DT.strftime('%Y-%m-%d')
SCENE_DOY = int(SCENE_DT.strftime('%j'))
SCENE_0UTC_DT = datetime.strptime(SCENE_DATE, '%Y-%m-%d')
SUN_ELEVATION = 64.2794
TEST_POINT = (-121.5265, 38.74)  # High ET field
# TEST_POINT = (-121.66, 38.90)  # Image centroid


# # Should these be test fixtures instead?
# # I'm not sure how to make them fixtures and allow input parameters
# def l8_image(ultra_blue=0.2, blue=0.2, green=0.2, red=0.2, nir=0.7,
#              swir1=0.2, swir2=0.2, bt=300):
#     """Construct a fake Landsat 8 SR image with renamed bands"""
#     return ee.Image.constant([ultra_blue, blue, green, red, nir, swir1, swir2, bt]) \
#         .rename(['ultra_blue', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'lst']) \
#         .set({
#             'system:index': SCENE_ID,
#             'system:time_start': SCENE_TIME,
#             'system:id': IMAGE_ID,
#             'SUN_ELEVATION': SUN_ELEVATION,
#         })


# def l7_image(blue=0.2, green=0.2, red=0.2, nir=0.7,
#              swir1=0.2, swir2=0.2, bt=300):
#     """Construct a fake Landsat 8 SR image with renamed bands"""
#     return ee.Image.constant([blue, green, red, nir, swir1, swir2, bt]) \
#         .rename(['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'lst']) \
#         .set({
#             'system:index': SCENE_ID,
#             'system:time_start': SCENE_TIME,
#             'system:id': IMAGE_ID,
#             'SUN_ELEVATION': SUN_ELEVATION,
#         })


def default_image(albedo=0.2, emissivity=0.964, lst=300, ndvi=0.5, ndwi=-0.1, savi=0.5):
    # First construct a fake 'prepped' input image
    # Map the values to an actual Landsat image
    return (
        ee.Image(IMAGE_ID)
        .select([0, 1, 2, 3, 4, 5])
        .multiply(0).double()
        .add(ee.Image.constant([albedo, emissivity, lst, ndvi, ndwi, savi]))
        .rename(['albedo', 'emissivity', 'lst', 'ndvi', 'ndwi', 'savi'])
        .set({
            'system:index': SCENE_ID,
            'system:time_start': SCENE_TIME,
            'system:id': IMAGE_ID,
            'SUN_ELEVATION': SUN_ELEVATION,
        })
    )


# Setting etr_source and etr_band on the default image to simplify testing
#   but these do not have defaults in the Image class init
def default_image_args(
        albedo=0.2,
        emissivity=0.964,
        lst=300,
        ndvi=0.5,
        ndwi=-0.1,
        savi=0.5,
        # meteorology_source_inst='NASA/NLDAS/FORA0125_H002',
        # meteorology_source_daily='IDAHO_EPSCOR/GRIDMET',
        # elevation_source='USGS/SRTMGL1_003',
        ndvi_cold=5,
        ndvi_hot=10,
        lst_cold=20,
        lst_hot=20,
        et_reference_source=15,
        et_reference_band='etr',
        et_reference_factor=0.85,
        et_reference_resample='nearest',
        ):
    return {
        'image': default_image(
            albedo=albedo, emissivity=emissivity, lst=lst, ndvi=ndvi, ndwi=ndwi, savi=savi
        ),
        # 'meteorology_source_inst': meteorology_source_inst,
        # 'meteorology_source_daily': meteorology_source_daily,
        # 'elevation_source': elevation_source,
        'ndvi_cold': ndvi_cold,
        'ndvi_hot': ndvi_hot,
        'lst_cold': lst_cold,
        'lst_hot': lst_hot,
        'et_reference_source': et_reference_source,
        'et_reference_band': et_reference_band,
        'et_reference_factor': et_reference_factor,
        'et_reference_resample': et_reference_resample,
    }


def default_image_obj(
        albedo=0.2,
        emissivity=0.964,
        lst=300,
        ndvi=0.5,
        ndwi=-0.1,
        savi=0.5,
        # meteorology_source_inst='NASA/NLDAS/FORA0125_H002',
        # meteorology_source_daily='IDAHO_EPSCOR/GRIDMET',
        # elevation_source='USGS/SRTMGL1_003',
        ndvi_cold=5,
        ndvi_hot=10,
        lst_cold=20,
        lst_hot=10,
        et_reference_source=15,
        et_reference_band='etr',
        et_reference_factor=0.85,
        et_reference_resample='nearest',
        ):
    return model.Image(**default_image_args(
        albedo=albedo,
        emissivity=emissivity,
        lst=lst,
        ndvi=ndvi,
        ndwi=ndwi,
        savi=savi,
        # meteorology_source_inst=meteorology_source_inst,
        # meteorology_source_daily=meteorology_source_daily,
        # elevation_source=elevation_source,
        ndvi_cold=ndvi_cold,
        ndvi_hot=ndvi_hot,
        lst_cold=lst_cold,
        lst_hot=lst_hot,
        et_reference_source=et_reference_source,
        et_reference_band=et_reference_band,
        et_reference_factor=et_reference_factor,
        et_reference_resample=et_reference_resample,
    ))


def test_Image_init_default_parameters():
    m = model.Image(default_image())
    assert m._meteorology_source_inst == 'NASA/NLDAS/FORA0125_H002'
    assert m._meteorology_source_daily == 'IDAHO_EPSCOR/GRIDMET'
    assert m._elevation_source == 'USGS/SRTMGL1_003'
    assert m._ndvi_cold == 5
    assert m._ndvi_hot == 10
    assert m._lst_cold == 20
    assert m._lst_hot == 20
    # assert m._et_reference_source == None
    # assert m._et_reference_band == None
    # assert m._et_reference_factor == None
    # assert m._et_reference_resample == None


# CGM - I should write image properties and image bands fixtures
def test_Image_ndvi_properties():
    """Test band name and if properties are set on the image"""
    output = utils.getinfo(default_image_obj().ndvi)
    assert output['bands'][0]['id'] == 'ndvi'
    assert output['properties']['system:index'] == SCENE_ID
    assert output['properties']['system:time_start'] == SCENE_TIME
    assert output['properties']['image_id'] == IMAGE_ID


def test_Image_ndvi_defaults(expected=0.1, tol=0.001):
    output = utils.point_image_value(
        ee.Image(default_image_obj(ndvi=expected).ndvi), TEST_POINT, 30)
    assert abs(output['ndvi'] - expected) <= tol


def test_Image_ndvi_values():
    output = utils.point_image_value(model.Image.from_image_id(IMAGE_ID).ndvi, TEST_POINT, 30)
    assert output['ndvi'] > 0


def test_Image_elevation_band_name():
    output = utils.getinfo(model.Image(default_image()).elevation)['bands'][0]['id']
    assert output == 'elevation'


@pytest.mark.parametrize(
    'source, xy, expected',
    [
        ['USGS/SRTMGL1_003', TEST_POINT, 3],
        ['USGS/SRTMGL1_003', [-120.0, 39.0], 1898],
        ['2364.351', TEST_POINT, 2364.35],
        [2364.351, TEST_POINT, 2364.35],
    ]
)
def test_Image_elevation_source_values(source, xy, expected, tol=0.01):
    output = utils.point_image_value(
        model.Image(default_image(), elevation_source=source).elevation, xy
    )
    assert abs(output['elevation'] - expected) <= tol


def test_Image_elevation_source_exception():
    with pytest.raises(ValueError):
        utils.getinfo(model.Image(default_image(), elevation_source=None).elevation)


def test_Image_et_reference_properties():
    """Test if properties are set on the reference ET image"""
    output = utils.getinfo(default_image_obj().et_reference)
    assert output['bands'][0]['id'] == 'et_reference'
    assert output['properties']['system:index'] == SCENE_ID
    assert output['properties']['system:time_start'] == SCENE_TIME
    assert output['properties']['image_id'] == IMAGE_ID


@pytest.mark.parametrize(
    'source, band, factor, xy, expected',
    [
        ['IDAHO_EPSCOR/GRIDMET', 'etr', 1, TEST_POINT, 11.2],
        ['IDAHO_EPSCOR/GRIDMET', 'etr', 0.85, TEST_POINT, 11.2 * 0.85],
        [
            'projects/openet/assets/reference_et/california/cimis/daily/v1',
            'etr', 1, TEST_POINT, 10.174
        ],
        [10, 'FOO', 1, TEST_POINT, 10.0],
        [10, 'FOO', 0.85, TEST_POINT, 8.5],
    ]
)
def test_Image_et_reference_sources(source, band, factor, xy, expected, tol=0.001):
    """Test getting reference ET values for a single date at a real point"""
    output = utils.point_image_value(default_image_obj(
        et_reference_source=source, et_reference_band=band,
        et_reference_factor=factor).et_reference, xy, 30)
    assert abs(output['et_reference'] - expected) <= tol


def test_Image_et_properties():
    """Test band name and if properties are set on the image"""
    output = utils.getinfo(default_image_obj().et)
    assert output['bands'][0]['id'] == 'et'
    assert output['properties']['system:index'] == SCENE_ID
    assert output['properties']['system:time_start'] == SCENE_TIME
    assert output['properties']['image_id'] == IMAGE_ID


# CGM - This test probably won't work since running the model with an image
#   that is constant is not going to work
# def test_Image_et_defaults(expected=0, tol=0.001):
#     output = utils.point_image_value(ee.Image(default_image_obj().et), TEST_POINT)
#     assert abs(output['et'] - expected) <= tol


def test_Image_et_fraction_properties():
    """Test if properties are set on the ET fraction image"""
    output =  utils.getinfo(default_image_obj().et_fraction)
    assert output['bands'][0]['id'] == 'et_fraction'
    assert output['properties']['system:index'] == SCENE_ID
    assert output['properties']['system:time_start'] == SCENE_TIME
    assert output['properties']['image_id'] == IMAGE_ID


# # CGM - Not sure why this one doesn't run
# def test_Image_et_fraction_values():
#     output = utils.point_image_value(model.Image.from_image_id(
#         IMAGE_ID,
#         et_reference_soruce='projects/openet/assets/reference_et/california/cimis/daily/v1',
#         et_reference_band='eto', et_reference_factor=1.0
#     ).et_fraction, TEST_POINT, 30)
#     assert output['et_fraction'] > 0


def test_Image_mask_properties():
    """Test if properties are set on the time image"""
    output = utils.getinfo(default_image_obj().mask)
    assert output['bands'][0]['id'] == 'mask'
    assert output['properties']['system:index'] == SCENE_ID
    assert output['properties']['system:time_start'] == SCENE_TIME
    assert output['properties']['image_id'] == IMAGE_ID


def test_Image_mask_values():
    output = utils.point_image_value(
        model.Image.from_image_id(IMAGE_ID).mask, TEST_POINT, 30
    )
    assert output['mask'] > 0


def test_Image_time_properties():
    """Test if properties are set on the time image"""
    output = utils.getinfo(default_image_obj().time)
    assert output['bands'][0]['id'] == 'time'
    assert output['properties']['system:index'] == SCENE_ID
    assert output['properties']['system:time_start'] == SCENE_TIME
    assert output['properties']['image_id'] == IMAGE_ID


def test_Image_time_values():
    output = utils.point_image_value(
        model.Image.from_image_id(IMAGE_ID).time, TEST_POINT, 30
    )
    assert output['time'] > 0


def test_Image_calculate_properties():
    """Test if properties are set on the output image"""
    output =  utils.getinfo(default_image_obj().calculate(['ndvi']))
    assert output['properties']['system:index'] == SCENE_ID
    assert output['properties']['system:time_start'] == SCENE_TIME
    assert output['properties']['image_id'] == IMAGE_ID


def test_Image_calculate_variables_default():
    output = utils.getinfo(default_image_obj().calculate())
    assert set([x['id'] for x in output['bands']]) == {'ndvi', 'lst', 'et', 'et_fraction'}
    # assert set([x['id'] for x in output['bands']]) == {'et'}


def test_Image_calculate_variables_custom():
    variables = {'ndvi'}
    output = utils.getinfo(default_image_obj().calculate(variables))
    assert set([x['id'] for x in output['bands']]) == variables


def test_Image_calculate_variables_all():
    variables = {'et', 'et_fraction', 'lst', 'ndvi', 'mask', 'time'}
    # variables = {'et', 'et_fraction', 'et_reference', 'mask', 'ndvi', 'time'}
    output = utils.getinfo(default_image_obj().calculate(variables=list(variables)))
    assert set([x['id'] for x in output['bands']]) == variables


# CGM - Figure out why test fails
# def test_Image_calculate_values():
#     """Test if the calculate method returns values"""
#     output_img = default_image_obj().calculate(['et'])
#     # output_img = default_image_obj().calculate(['et', 'et_reference', 'et_fraction'])
#     assert utils.constant_image_value(output_img.select(['et']))['et'] > 0
#     # assert utils.constant_image_value(output_img.select(['et_reference']))['et_reference'] > 0
#     # assert utils.constant_image_value(output_img.select(['et_fraction']))['et_fraction'] > 0


def test_Image_calculate_variables_valueerror():
    """Test if calculate method raises a valueerror for invalid variables"""
    with pytest.raises(ValueError):
        utils.getinfo(default_image_obj().calculate(['FOO']))


def test_Image_from_landsat_c2_sr_default_image():
    """Test that the classmethod is returning a class object"""
    output = model.Image.from_landsat_c2_sr(
        ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_038031_20130828'))
    assert type(output) == type(default_image_obj())


@pytest.mark.parametrize(
    'image_id',
    [
        # 'LANDSAT/LT04/C02/T1_L2/LT04_044033_19830812',
        'LANDSAT/LT05/C02/T1_L2/LT05_044033_20110716',
        'LANDSAT/LE07/C02/T1_L2/LE07_044033_20170708',
        'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
        'LANDSAT/LC09/C02/T1_L2/LC09_044033_20220127',
    ]
)
def test_Image_from_landsat_c2_sr_landsat_image(image_id):
    """Test instantiating the class from a real Landsat images"""
    output = utils.getinfo(model.Image.from_landsat_c2_sr(ee.Image(image_id)).ndvi)
    assert output['properties']['system:index'] == image_id.split('/')[-1]


# def test_Image_from_landsat_c2_sr_exception():
#     """Test instantiating the class for an invalid image ID"""
#     with pytest.raises(Exception):
#         model.Image.from_landsat_c2_sr(ee.Image('FOO'))._index.getInfo()


def test_Image_from_landsat_c2_sr_c2_lst_correct_arg():
    """Test if the c2_lst_correct parameter can be set (not if it works)"""
    image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_031034_20160702'
    output = model.Image.from_landsat_c2_sr(image_id, c2_lst_correct=True)
    assert type(output) == type(default_image_obj())


def test_Image_from_landsat_c2_sr_c2_lst_correct_fill():
    """Test if the c2_lst_correct fills the LST holes in Nebraska"""
    image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_031034_20160702'
    xy = (-102.08284, 37.81728)
    # CGM - Is the uncorrected test needed?
    uncorrected = utils.point_image_value(
        model.Image.from_landsat_c2_sr(image_id, c2_lst_correct=False).lst, xy)
    assert uncorrected['lst'] is None
    corrected = utils.point_image_value(
        model.Image.from_landsat_c2_sr(image_id, c2_lst_correct=True).lst, xy)
    assert corrected['lst'] > 0
    # # Exact test values copied from openet-core
    # assert abs(corrected['lst'] - 306.83) <= 0.25


def test_Image_from_landsat_c2_sr_lst_source_arg():
    """Test if the lst_source parameter can be set (not if it works)"""
    image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_031035_20160702'
    output = model.Image.from_landsat_c2_sr(
        image_id, lst_source='projects/openet/assets/lst/landsat/c02')
    assert type(output) == type(default_image_obj())


def test_Image_from_landsat_c2_sr_lst_source_values():
    """Test if the lst_source image can be read"""
    image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_031035_20160702'
    xy = (-102.4, 36.1)
    lst_source = 'projects/openet/assets/lst/landsat/c02'
    output_img = model.Image.from_landsat_c2_sr(image_id, lst_source=lst_source).lst
    output = utils.point_image_value(output_img, xy)
    assert abs(output['lst'] - 322.8) <= 0.1
    # assert output_img.get('lst_source_id').getInfo().startswith(lst_source)


@pytest.mark.parametrize(
    'image_id',
    [
        'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
    ]
)
def test_Image_from_image_id(image_id):
    """Test instantiating the class using the from_image_id method"""
    output = utils.getinfo(model.Image.from_image_id(image_id).ndvi)
    assert output['properties']['system:index'] == image_id.split('/')[-1]
    assert output['properties']['image_id'] == image_id


def test_Image_from_method_kwargs():
    """Test that the init parameters can be passed through the helper methods"""
    assert model.Image.from_landsat_c2_sr(
        'LANDSAT/LC08/C02/T1_L2/LC08_042035_20150713',
        elevation_source='FOO')._elevation_source == 'FOO'


@pytest.mark.parametrize(
    'image_id, xy, albedo, emissivity, lst, ndvi, ndwi, savi',
    [
        [
            'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', (-121.5265, 38.74),
            0.0991, 0.9800, 305.9179, 0.8472, -0.7780, 0.4631
        ],
        [
            'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', (-121.640, 38.788),
            0.1828, 0.9517, 328.2546, 0.1336, -0.2401, 0.0898
        ],

    ]
)
def test_Image_from_landsat_c2_sr_image_values(
        image_id, xy, albedo, emissivity, lst, ndvi, ndwi, savi, tol=0.0001
):
    """Check the input image values at specific test point locations"""
    image = model.Image.from_landsat_c2_sr(image_id)
    assert abs(utils.point_image_value(image.albedo, xy)['albedo'] - albedo) <= tol
    assert abs(utils.point_image_value(image.emissivity, xy)['emissivity'] - emissivity) <= tol
    assert abs(utils.point_image_value(image.lst, xy)['lst'] - lst) <= tol
    assert abs(utils.point_image_value(image.ndvi, xy)['ndvi'] - ndvi) <= tol
    assert abs(utils.point_image_value(image.ndwi, xy)['ndwi'] - ndwi) <= tol
    assert abs(utils.point_image_value(image.savi, xy)['savi'] - savi) <= tol


def test_Image_et_values():
    output = utils.point_image_value(model.Image.from_image_id(IMAGE_ID).et, TEST_POINT, 30)
    assert output['et'] > 0


@pytest.mark.parametrize(
    'image_id, xy, flag_g, et',
    [
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', (-121.5265, 38.74), 'default', 7.1812],
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', (-121.640, 38.788), 'default', 3.8362],
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', (-121.5265, 38.74), 'g_brazil', 7.2335],
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', (-121.640, 38.788), 'g_brazil', 3.0552],
    ]
)
def test_Image_from_landsat_c2_sr_et_values(image_id, xy, et, flag_g, tol=0.0001):
    """Check the output ET values at specific test point locations"""
    image = model.Image.from_landsat_c2_sr(image_id, flag_g=flag_g)
    assert abs(utils.point_image_value(image.et, xy)['et'] - et) <= tol
