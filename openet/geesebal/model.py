import math

import ee

from openet.geesebal import utils

DEG2RAD = math.pi / 180.0


# TODO: savi variable is not currently used but is being read from image instead
#   The image input could probably be dropped if sun_elevation was passed in instead
def et(
    ndvi,
    ndwi,
    lst,
    albedo,
    emissivity,
    savi,
    tmin,
    tmax,
    tair,
    ux,
    rh,
    rso_inst,
    rso_24h,
    tfac,
    elevation,
    ndvi_cold,
    ndvi_hot,
    lst_cold,
    lst_hot,
    sun_elevation,
    time_start,
    geometry_image,
    proj,
    coords,
    flag_g='g_brazil'
):
    """
    Daily Evapotranspiration [mm day-1].

    Parameters
    ----------
    ndvi : ee.Image
        Normalized difference vegetation index.
    ndwi : ee.Image
        Normalized difference water index.
    lst : ee.Image
        Land Surface Temperature [K].
    albedo : ee.Image
        Surface albedo.
    emissivity : ee.Image
        Broad-band surface emissivity.
    savi : ee.Image
        Soil-adjusted vegetation index.
    tmin : ee.Image
        Minimum daily air temperature [K].
    tmax : ee.Image
        Maximum daily air temperature [K].
    tair : ee.Image
        Instantaneous air temperature [C].
    ux : ee.Image
        Wind speed [m/s].
    rh : ee.Image
        Relative humidity (%).
    rso_inst : ee.Image
        Instantaneous incoming shortwave radiation [W m-2].
    rso_24h : ee.Image
        Daily incoming shortwave radiation [W m-2].
    tfac : ee.Image
        Temperature adjustment offset.
    elevation : ee.Image
        Elevation [m].
    ndvi_cold : ee.Number, int
        NDVI Percentile value to determinate cold pixel.
    ndvi_hot : ee.Number, int
        NDVI Percentile value to determinate hot pixel.
    lst_cold : ee.Number, int
        Lst Percentile value to determinate cold pixel.
    lst_hot : ee.Number, int
        Lst Percentile value to determinate hot pixel.
    sun_elevation : ee.Number, float
        Sun elevation angle read from Landsat SUN_ELEVATION property [deg].
    time_start : str
        Image property: time start of the image.
    geometry_image : ee.Geometry
        Image geometry.
    proj : ee.Image
        Landsat image projection.
    coords : ee.Image
        Landsat image Latitude and longitude.
    flag_g: str (optional)
        Geographic identifier. 
        Options: 'default' to global and 'g_brazil' to Brazil.

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Laipelt2021] L. Laipelt, R. Kayser, A. Fleischmann, A. Ruhoff,
        W. Bastiaanssen, T. Erickson, F. Melton,
        Long-term monitoring of evapotranspiration using the SEBAL
        algorithm and Google Earth Engine cloud computing,
        ISPRS Journal of Photogrammetry and Remote Sensing, Vol 178,
        https://doi.org/10.1016/j.isprsjprs.2021.05.018
    """

    # Image properties
    date = ee.Date(time_start)
    year = ee.Number(date.get('year'))

    # Endmembers
    top_ndvi = ee.Number(ndvi_cold)
    coldest_lst = ee.Number(lst_cold)
    
    # Terrain cos
    cos_zn = cos_terrain(elevation, time_start, coords)

    # LL: iterative process or endmembers selections may return empty values
    # in this case, return an empty image instead of broken the code
    try:
        # Air temperature correction [K]
        tair_dem = tair_dem_correction(tmin, tmax, elevation)

        # Land surface temperature correction [K]
        lst_dem = lst_correction(lst, elevation, tair_dem, rh, sun_elevation, time_start, coords)
        
        # Instantaneous net radiation using reanalysis dataset [W m-2]
        rad_inst = radiation_inst(
            lst, emissivity, albedo, tair, rh, elevation, rso_inst, sun_elevation, cos_zn
        )
        
        # Instantaneous soil heat flux [W m-2]
        g_inst = soil_heat_flux(rad_inst, ndvi, albedo, lst, ndwi, flag_g, year)
        
        # Daily ney radiation [W m-2]
        rad_24h = radiation_24h(time_start, tmax, tmin, elevation, rso_24h)

        # Cold pixels
        cold_pixels = cold_pixel(
            ndvi, ndwi, lst_dem, top_ndvi, coldest_lst, geometry_image, coords, elevation, proj
        )

        n_Ts_cold = ee.Number(cold_pixels.get('temp'))
        
        # Hot pixels
        hot_pixels = hot_pixel(
            ndvi, ndwi, lst_dem, rad_inst, g_inst, ndvi_hot, lst_hot,
            geometry_image, coords, tfac, elevation, proj
        )
        
        # Instantaneous sensible heat flux [W m-2]
        h_inst = sensible_heat_flux(lst_dem, savi, ux, n_Ts_cold, hot_pixels, elevation)
        
        # Daily evapotranspiration [mm day-1]
        et_24hr = daily_et(h_inst, g_inst, rad_inst, lst_dem, rad_24h)

    except Exception as e:
        # CGM - We should probably log the exception so the user knows,
        #   but this will cause problems when mapping over a collection
        print(f'Unhandled Exception: {e}')

        # Return a masked image
        et_24hr = ee.Image.constant(0).updateMask(0).rename('et')

    return et_24hr


def tao_sw(elev, tair, rh, cos_z0):
    """
    Correct declivity and aspect effects from Land Surface Temperature.

    Parameters
    ----------
    elev : ee.Image
        Elevation [m].
    tair : ee.Image
        Air temperature [Celsius].
    rh : ee.Image
        Relative Humidity [%]
    cos_z0 : ee.Number, ee.Image
        Solar zenith angle cos.

    Returns
    -------
    ee.Image
    """
    # Atmospheric pressure [kPa] (FAO56 Eqn 7)
    pres = elev.expression('101.3 * ((293 - (0.0065 * Z)) / 293) ** 5.26 ', {'Z': elev})

    # Saturated vapor pressure [kPa] (FAO56 Eqn 11)
    es = tair.expression('0.6108 * exp((17.27 * tair) / (tair + 237.3))', {'tair': tair})

    # Actual vapor pressure [kPa]  (FAO56 Eqn 10)
    ea = es.multiply(rh).divide(100).rename('ea')

    # Water in the atmosphere [mm] (Garrison and Adler (1990))
    w = ea.expression('(0.14 * EA * PATM) + 2.1', {'PATM': pres, 'EA': ea})

    # Broad-band atmospheric transmissivity (ASCE-EWRI (2005))
    tao_sw_img = pres.expression(
        '0.35 + 0.627 * exp(((-0.00146 * P) / (Kt * ct)) - (0.075 * (W / ct) ** 0.4))',
        {'P': pres, 'W': w, 'Kt': 1.0, 'ct': cos_z0},
    )

    return tao_sw_img.rename('tao_sw')


def cos_terrain(elev, time_start, coords):
    """
    Cosine zenith angle elevation (Allen et al. (2006)).

    Parameters
    ----------
    elev : ee.Image
        Elevation [m].
    time_start : str
        Image property: time start of the image.
    coords : ee.Image
        Latitude and longitude of the image.

    Returns
    -------
    ee.Image
    """

    # Day of the year
    doy = ee.Date(time_start).getRelative('day', 'year').add(1)
    hour = ee.Number(ee.Date(time_start).get('hour'))
    minutes = ee.Number(ee.Date(time_start).get('minutes'))

    # Slope and aspect
    slope_aspect = ee.Terrain.products(elev)

    # Variables
    B = doy.subtract(81).multiply(360 / 365)
    delta = ee.Number(23.45 * DEG2RAD).sin().multiply(B.multiply(DEG2RAD).sin()).asin()
    s = slope_aspect.select('slope').multiply(DEG2RAD)
    gamma = slope_aspect.select('aspect').subtract(180).multiply(DEG2RAD)
    phi = ee.Image.pixelLonLat().select('latitude').multiply(DEG2RAD)

    # Constants
    delta = ee.Image(delta)
    a = delta.sin().multiply(phi.cos()).multiply(s.sin()).multiply(gamma.cos())\
        .subtract(delta.sin().multiply(phi.sin().multiply(s.cos())))
    b = delta.cos().multiply(phi.cos()).multiply(s.cos())\
        .add(delta.cos().multiply(phi.sin().multiply(s.sin()).multiply(gamma.cos())))
    c = delta.cos().multiply(s.sin()).multiply(gamma.sin())

    # Centroid image
    longitude_center = coords.select('longitude')

    delta_gtm = longitude_center.divide(ee.Image(15)).int()

    # Local hour time
    lht = delta_gtm.add(hour).add(minutes.divide(60))

    w = lht.subtract(12).multiply(15).multiply(DEG2RAD)

    # Cosine zenith angle elevation
    cos_zn = w.expression(
        '-a + b * w_cos + c * w_sin',
        {'a': a, 'b': b, 'c': c, 'w_cos': w.cos(), 'w_sin': w.sin()},
    )

    return cos_zn


def tair_dem_correction(tmin, tmax, elev):
    """
    Correct air temperature for mountain areas 

    Parameters
    ----------
    tmin : ee.Image
        Minimum temperature [Celsius].
    tmax : ee.Image
        Maximum temperature [Celsius].
    elev : ee.Image
        Elevation [m].

    Returns
    -------
    ee.Image
    """

    tmin_dem = tmin.expression(
        'temp - 0.0065 * (dem - alt_meteo)', 
        {'temp': tmin, 'dem': elev, 'alt_meteo': ee.Number(2)}
    )

    tmax_dem = tmax.expression(
        'temp - 0.0065 * (dem - alt_meteo)', 
        {'temp': tmax, 'dem': elev, 'alt_meteo': ee.Number(2)}
    )

    tair_dem = tmin_dem.add(tmax_dem).divide(2)

    return tair_dem.rename('tair_dem')


def lst_correction(lst, elev, tair, rh, sun_elevation, time_start, coords):
    """
    Correct declivity and aspect effects from Land Surface Temperature.

    Parameters
    ----------
    lst : ee.Image
        Land surface temperature [K]
    elev : ee.Image
        Elevation [m].
    tair : ee.Image
        Air temperature [Celsius].
    rh : ee.Image
        Relative Humidity [%].
    sun_elevation : ee.Number, ee.Image
        Sun elevation angle.
    time_start : str
        Image property: time start of the image.
    coords : ee.Image
        Landsat image Latitude and longitude.

    Returns
    -------
    ee.Image

    References
    ----------
    Jaafar, H.H., Ahmad, F.A., 2019. Time series trends of Landsat-based ET using automated
    calibration in METRIC and SEBAL: The Bekaa Valley, Lebanon. Remote Sens.
    Environ. https://doi.org/https://doi.org/10.1016/j.rse.2018.12.033.
    """

    # Solar constant [W m-2]
    gsc = ee.Image.constant(1367)

    # Day of the year
    doy = ee.Date(time_start).getRelative('day', 'year').add(1)

    # Inverse relative distance earth-sun (FAO56 Eqn 23)
    dr = doy.multiply(2 * math.pi / 365).cos().multiply(0.033).add(1)

    # Atmospheric pressure [kPa] (FAO56 Eqn 7)
    pres = lst.expression('101.3 * ((293 - (0.0065 * Z)) / 293) ** 5.26 ', {'Z': elev})

    # Solar zenith angle over a horizontal surface
    solar_zenith = ee.Number(90).subtract(sun_elevation)

    solar_zenith_radians = solar_zenith.multiply(DEG2RAD)
    cos_theta = solar_zenith_radians.cos()

    # Air density [Kg m-3]
    air_dens = lst.expression('(1000 * Pair) / (1.01 * LST * 287)', {'Pair': pres, 'LST': lst})

    # Temperature lapse rate [K m-1]
    temp_lapse_rate = ee.Number(0.0065)

    # Correcting land surface temperature [K] with temperature lapse rate
    temp_corr = lst.add(elev.multiply(temp_lapse_rate))

    cos_zn = cos_terrain(elev, time_start, coords)

    # Broad-band atmospheric transmissivity (ASCE-EWRI (2005))
    tao_sw_img = tao_sw(elev, tair, rh, cos_zn)

    # Corrected Land Surface temperature [K] (Zaafar and Farah (2020) Eqn 2)
    lst_dem = lst.expression(
        'Temp_corr + (Gsc * dr * Transm_corr * cos_zn - Gsc * dr * Transm_corr * cos_zenith_flat) / '
        '(air_dens * 1004 * 0.050)',
        {
            'Temp_corr': temp_corr,
            'Gsc': gsc,
            'dr': dr,
            'Transm_corr': tao_sw_img,
            'cos_zenith_flat': cos_theta,
            'cos_zn': cos_zn,
            'air_dens': air_dens},
    )

    return lst_dem.rename('lst_dem')


def homogeneous_mask(ndvi, proj):
    """
    Homogeneous mask for endmembers selection (Allen et al. (2013)).

    Parameters
    ----------
    ndvi : ee.Image
        Normalized difference vegetation index (ndvi).
    proj : ee.Dictionary
        Landsat image projection.

    Returns
    -------
    ee.Image
    """

    # Calculate NDVI standard deviation in pixel neighborhood
    sd_ndvi = (
        ndvi.reduceNeighborhood(
            reducer=ee.Reducer.stdDev(), 
            kernel=ee.Kernel.square(radius=3, units='pixels'), skipMasked=False
        )
        .reproject(proj)
        .updateMask(1)
    )
    
    # Calculate mean NDVI in pixel neighborhood
    mean_ndvi = (
        ndvi.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.square(radius=3, units='pixels'), skipMasked=False
        )
        .reproject(proj)
        .updateMask(1)
    )

    cv_mask = sd_ndvi.divide(mean_ndvi).lte(0.15).selfMask()

    return ee.Image(cv_mask)


def slope_mask(elev, proj):
    """
    Applies a filter for slope values in the image, allowing only pixels from flat 
    regions to be selected as endmembers

    Parameters
    ----------
    elev : ee.Image
        Elevation [m].
    proj : ee.Dictionary
        Landsat image projection.

    Returns
    -------
    ee.Image
    """

    dem_slope = (
        ee.Terrain.slope(elev).divide(180).multiply(math.pi).tan().multiply(100)
        .reproject(proj)
    )
    slp_mask = dem_slope.lte(30).selfMask()    

    return ee.Image(slp_mask)


def radiation_inst(lst, emissivity, albedo, tair, rh, elev, swdown_inst, sun_elevation, cos_zn):
    """
    Instantaneous Net Radiation [W m-2]

    Parameters
    ----------
    lst : ee.Image
        Land surface temperature [k].
    emissivity : ee.Image
        Emissivity.
    albedo : ee.Image
        Albedo.
    tair : ee.Image
        Air temperature [Celsius].
    rh : ee.Image
        Relative Humidity [%].
    elev : ee.Image
        Elevation [m].
    swdown_inst : ee.Image
        Instantaneous Short Wave radiation [W m-2].
    sun_elevation : ee.Number, int
        Sun elevation information.
    cos_zn : ee.Image
        Solar zenith angle cos (aspect/slope).

    Returns
    -------
    ee.Image
    """

    # Up long wave radiation [W m-2]
    rad_long_up = lst.expression('emi * 5.67e-8 * (lst ** 4)', {'emi': emissivity, 'lst': lst})

    # Transmissivity 
    tao_sw_img = tao_sw(elev, tair, rh, cos_zn)

    # Down long wave radiation [W m-2] 
    rad_long_down = lst.expression(
        '(0.85 * (- log_taosw) ** 0.09) * 5.67e-8 * (n_Ts_cold ** 4)',
        {'log_taosw': tao_sw_img.log(), 'n_Ts_cold': tair.add(273.15)},
    )

    # Solar zenith
    solar_zenith = ee.Number(90).subtract(sun_elevation)
    solar_zenith_radians = solar_zenith.multiply(DEG2RAD)
    cos_zeni = solar_zenith_radians.cos()

    # Obtaining a Short down wave radiation correcting mountainous effect 
    swdown_inst_dem = swdown_inst.multiply(cos_zn.divide(cos_zeni))

    # Net radiation [W m-2]
    rn_inst = lst.expression(
        '((1 - alfa) * Rs_down) + Rl_down - Rl_up - ((1 - e_0) * Rl_down)',
        {
            'alfa': albedo,
            'Rs_down': swdown_inst_dem,
            'Rl_down': rad_long_down,
            'Rl_up': rad_long_up,
            'e_0': emissivity,
        },
    )

    return rn_inst.rename('rn_inst')


def soil_heat_flux(rn, ndvi, albedo, lst, ndwi, flag_g, year):
    """
    Instantaneous Soil Heat Flux [W m-2]

    Parameters
    ----------
    rn : ee.Image
        Instantaneous Net Radiation [W m-2]
    ndvi : ee.Image
        Normalized difference vegetation index.
    albedo : ee.Image
        Albedo.
    lst : ee.Image
        Land surface temperature [K].
    ndwi : ee.Image
        Normalized difference water index.
    year: ee.Number, int
        Year.
    flag_g: str (optional)
        Geographic identifier. 
        Options: 'default' to global and 'g_brazil' to Brazil.

    Returns
    -------
    ee.Image
    """

    # Soil heat flux [W m-2] 
    g_crop = rn.expression(
        'rn * (lst - 273.15) * (0.0038 + (0.0074 * albedo)) * (1 - 0.98 * (ndvi ** 4))',
        {'rn': rn, 'ndvi': ndvi, 'albedo': albedo, 'lst': lst},
    )

    # G global
    if flag_g == 'default':
        g = g_crop

    # G for Brazil
    elif flag_g == 'g_brazil':
        # Soil heat flux [W m-2] for all classes except croplands (Ruhoff, 2019)
        g_noncrop = rn.expression(
            'rn * lst * (0.015 * albedo) * (1 - 0.8 * (ndvi ** (1/3)))',
            {'rn': rn, 'ndvi': ndvi, 'albedo': albedo, 'lst': lst},
        )

        # TODO: Move landuse processing out of model and into Image,
        #   Pass in crop or noncrop mask similar to PTJPL, SSEBop, RADET
        # MapBiomas classification for the image year
        year_min = 1985
        year_max = 2025
        year_condition = ee.Number(year).max(year_min).min(year_max)
        band = ee.String('classification_').cat(year_condition.format())
        lc = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3').select(band)
        map_classes = ee.List([1, 3, 4, 5, 6, 49, 10, 11, 12, 32, 29, 50, 14, 15, 18, 19, 39, 20, 40, 62, 41, 36, 46, 47, 35, 48, 9, 21, 22, 23, 24, 30, 75, 25, 26, 33, 31, 27])
        new_classes = ee.List([1, 1, 2, 0, 0,  0,  0,  0,  3,  0,  0,  0,  0,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
        lulc = lc.select(band).remap(map_classes, new_classes)
        crop = lulc.eq(4)

        g = g_noncrop.where(crop, g_crop)

    else:
        raise ValueError(f"Invalid flag_g: {flag_g}. Use 'default' or 'g_brazil'.")

    # Considering G as 50% of net radiation for water bodies
    g = g.where(ndwi.gt(0), rn.multiply(0.5))

    return g.rename('g_inst')


def radiation_24h(time_start, tmax, tmin, elev, rso_24h):
    """
    Daily Net radiation [W m-2] - FAO56

    Parameters
    ----------
    time_start : ee.Date
        Date information of the image.
    tmax : ee.Image
        Maximum air temperature [Celsius].
    tmin : ee.Image
        Minimum air temperature [Celsius].
    elev : ee.Image
        Digital Elevation information [m].
    rso_24h : ee.Image
        Daily Short wave radiation [W m-2]

    Returns
    -------
    ee.Image

    References
    ----------
    .. [FAO56] Allen, R., Pereira, L., Raes, D., & Smith, M. (1998).
       Crop evapotranspiration: Guidelines for computing crop water
       requirements. FAO Irrigation and Drainage Paper (Vol. 56).
    """

    # Convert to MJ m-2
    rs = rso_24h.multiply(0.0864).rename('Rs')

    # Solar constant [MJ m-2]
    gsc = 0.0820

    # Day of the year
    doy = ee.Date(time_start).getRelative('day', 'year').add(1)

    # Inverse relative distance earth-sun (FAO56 Eqn 23)
    dr = tmax.expression('1 + (0.033 * cos((2 * pi / 365) * doy))', {'doy': doy, 'pi': math.pi})

    # Solar declination [rad] (FAO56 Eqn 24)
    sd = tmax.expression(
        '0.40928 * sin(((2 * pi / 365) * doy) - 1.39)', {'doy': doy, 'pi': math.pi}
    )

    # Latitude of the image
    lat = tmax.pixelLonLat().select(['latitude']).multiply(DEG2RAD).rename('latitude')

    #  Sunset hour angle [rad] (FAO56 Eqn 25)
    ws = tmax.expression('acos(-tan(Lat) * tan(Sd))', {'Lat': lat, 'Sd': sd})

    # Extraterrestrial radiation [MJ m-2 d-1] (FAO56 Eqn 21)
    rad_a = tmax.expression(
        'Ws * sin(Lat) * sin(Sd) + cos(Lat) * cos(Sd) * sin(Ws)',
        {'Ws': ws, 'Lat': lat, 'Sd': sd}
    )

    ra = tmax.expression(
        '((24 * 60) / pi) * Gsc * Dr * rad_a',
        {'pi': math.pi, 'Gsc': gsc, 'Dr': dr, 'rad_a': rad_a}
    )
    
    # Simplified clear sky solar formulation [MJ m-2 d-1] (FAO56 Eqn 37)
    rso = tmax.expression('(0.75 + 2E-5 * z) * Ra', {'z': elev, 'Ra': ra})

    # Net shortwave radiation [MJ m-2 d-1] (FAO56 Eqn 38)
    rns = tmax.expression('(1 - albedo) * Rs', {'Rs': rs, 'albedo': 0.23})

    # Actual vapor pressure [MJ m-2 d-1] (FAO56 Eqn 11)
    ea = tmax.expression('0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))', {'T_air': tmin})

    # Net longwave radiation [MJ m-2 d-1] (FAO56 Eqn 39)
    rnl = tmax.expression(
        '4.901E-9 * ((Tmax ** 4 + Tmin ** 4) / 2) * (0.34 - 0.14 * sqrt(ea)) * '
        '(1.35 * (Rs / Rso) - 0.35)',
        {'Tmax': tmax.add(273.15), 'Tmin': tmin.add(273.15), 'ea': ea, 'Rs': rs, 'Rso': rso},
    )

    # Net radiation [MJ m-2 d-1] (FAO56 Eqn 40)
    rn = tmax.expression('Rns - Rnl', {'Rns': rns, 'Rnl': rnl})

    # Convert to W m-2
    rn = rn.multiply(11.6)

    return rn.rename('rad_24h')


def cold_pixel(ndvi, ndwi, lst_dem, ndvi_cold, lst_cold, geometry_image, coords, elev, proj):
    """
    Simplified CIMEC method to select the cold pixel

    Parameters
    ----------
    ndvi : ee.Image
        Normalized difference vegetation index.
    ndwi : ee.Image
        Normalized difference water index.
    lst_dem : ee.Image
        Land surface temperature [K].
    ndvi_cold : ee.Number, int
        NDVI Percentile value to determinate cold pixel.
    lst_cold : ee.Number, int
        LST Percentile value to determinate cold pixel.
    geometry_image : ee.Geometry
        Landsat image geometry.
    coords : ee.Image
        Latitude and longitude coordinates of the image.
    elev : ee.Image
        Digital Elevation information [m].
    proj : ee.Dictionary
        Landsat image projection.

    Returns
    -------
    ee.Dictionary

    Notes
    -----
    Based on Allen et al (2013) procedure to represent extreme conditions
    to use in METRIC (adaptable for SEBAL) using endmembers candidates from
    pre-defined percentiles of LST and NDVI.

    References
    ----------
    ..[Allen2013] Allen, R.G., Burnett, B., Kramber, W., Huntington, J.,
        Kjaersgaard, J., Kilic, A., Kelly, C., Trezza, R., (2013).
        Automated Calibration of the METRIC-Landsat Evapotranspiration Process.
        JAWRA J. Am. Water Resour. Assoc. 49, 563–576.
    """

    # Filtering only positive ndvi values
    pos_ndvi = ndvi.updateMask(ndvi.gt(0)).rename('post_ndvi')

    # Creating a negative ndvi raster for filtering
    ndvi_neg = pos_ndvi.multiply(-1).rename('ndvi_neg')

    # Creating a negative lst raster for filtering
    lst_neg = lst_dem.multiply(-1).rename('lst_neg')

    # Lst for non water pixels 
    lst_nw = lst_dem.updateMask(ndwi.lte(0)).rename('lst_nw')

    # Homogeneous mask for ndvi
    stdev_ndvi = homogeneous_mask(ndvi, proj)

    # Flat areas mask 
    slope_filt = slope_mask(elev, proj)   
    
    # Creating a raster with all the parameters
    images = pos_ndvi.addBands([ndvi, ndvi_neg, pos_ndvi, lst_neg, lst_nw, coords])

    # Estimating ndvi percentile [coldest]
    # TODO: Why does this call get ndvi.geometry() while the others use geometry_image?
    #   Could we avoid passing in the geometry_image parameter?
    d_perc_top_NDVI = (
        images.select('ndvi_neg')
        .updateMask(stdev_ndvi)
        .updateMask(slope_filt)
        .reduceRegion(
            reducer=ee.Reducer.percentile([ndvi_cold]),
            geometry=ndvi.geometry(),
            scale=30,
            maxPixels=1e9,
        )
        .combine(ee.Dictionary({'ndvi_neg': 100}), False)
    )
   
    # Get top ndvi value
    n_perc_top_NDVI = ee.Number(d_perc_top_NDVI.get('ndvi_neg'))
   
    # Filtering ndvi raster with the percecilte
    i_top_NDVI = images.updateMask(images.select('ndvi_neg').lte(n_perc_top_NDVI))

    # Filtering lst
    d_perc_low_LST = (
        i_top_NDVI.select('lst_nw')
        .updateMask(stdev_ndvi)
        .updateMask(slope_filt)
        .reduceRegion(
            reducer=ee.Reducer.percentile([lst_cold]),
            geometry=geometry_image,
            scale=30,
            maxPixels=1e9,
        )
        .combine(ee.Dictionary({'lst_nw': 350}), False)
    )
          
    # Get low lst value
    n_perc_low_LST = ee.Number(d_perc_low_LST.get('lst_nw'))
   
    # Filtering lst raster with the percentile
    i_cold_lst = i_top_NDVI.updateMask(i_top_NDVI.select('lst_nw').lte(n_perc_low_LST))
 
    # Filters 
    c_lst_cold20 = i_cold_lst.updateMask(images.select('lst_nw').gte(200))
    c_lst_cold20_int = c_lst_cold20.select('lst_nw').min(1).max(1).int().rename('int')
    c_lst_cold20 = c_lst_cold20.addBands(c_lst_cold20_int)
  
    # Getting the number of pixels found
    sum_final_cold_pix = c_lst_cold20.select('int').reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry_image,
        scale=30,
        maxPixels=1e9,
    )
        
    n_sum_final_cold_pix = ee.Number(sum_final_cold_pix.get('int'))

    # Get Cold Pixels 
    # 5000 random pixels and select the 200 coldest
    def set_pt_geometry(f):
        return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))

    fc_cold_pix = (
        c_lst_cold20.stratifiedSample(5000, 'int', geometry_image, 30)
        .map(set_pt_geometry)
        .sort('lst_nw', True)  
        .limit(200)
    )  
 
    n_Ts_cold = ee.Number(fc_cold_pix.aggregate_mean('lst_nw'))
    n_long_cold = ee.Number(fc_cold_pix.aggregate_first('longitude'))
    n_lat_cold = ee.Number(fc_cold_pix.aggregate_first('latitude'))
    n_ndvi_cold = ee.Number(fc_cold_pix.aggregate_mean('post_ndvi'))

    # Dictionary
    d_cold_pixel = ee.Dictionary({
        'temp': n_Ts_cold,
        'ndvi': n_ndvi_cold,
        'x': n_long_cold,
        'y': n_lat_cold,
        'sum': n_sum_final_cold_pix,
    }).combine(ee.Dictionary({'temp': 0, 'ndvi': 0, 'x': 0, 'y': 0, 'sum': 0}), False)

    return d_cold_pixel


def hot_pixel(ndvi, ndwi, lst_dem, rn, g, ndvi_hot, lst_hot, geometry_image, coords, tfac, elev, proj): 
    """
    Simplified CIMEC method to select the hot pixel

    Parameters
    ----------
    ndvi : ee.Image
        Normalized difference vegetation index.
    ndwi : ee.Image
        Normalized difference water index.
    lst_dem : ee.Image
        Land surface temperature [K] corrected by altitude.
    rn : ee.Image
        Instantaneous Net Radiation [W m-2]
    g : ee.Image
        Instantaneous Soil heat flux [W m-2]
    ndvi_hot : ee.Number, int
        NDVI Percentile value to determinate hot pixel.
    lst_hot : ee.Number, int
        LST Percentile value to determinate hot pixel.
    geometry_image : ee.Geometry
        Image geometry.
    coords : ee.Image
        Latitude and longitude coordinates of the image.
    tfac : ee.Image
        Temperature adjustment offset [K].
    elev : ee.Image
        Elevation [m].
    proj : ee.Dictionary
        Landsat image projection.

    Returns
    -------
    ee.Dictionary

    Notes
    -----
    Based on Allen et al (2013) procedure to represent extreme conditions
    to use in METRIC (adaptable for SEBAL) using endmembers candidates from
    pre-defined percentiles of LST and NDVI.

    References
    ----------
    .. [Allen2013] Allen, R.G., Burnett, B., Kramber, W., Huntington, J.,
        Kjaersgaard, J., Kilic, A., Kelly, C., Trezza, R., (2013).
        Automated Calibration of the METRIC-Landsat Evapotranspiration Process.
        JAWRA J. Am. Water Resour. Assoc. 49, 563–576.
    """

    # Filtering only positive ndvi values
    pos_ndvi = ndvi.updateMask(ndvi.gt(0)).rename('post_ndvi')

    # Creating a negative ndvi raster for filtering
    ndvi_neg = pos_ndvi.multiply(-1).rename('ndvi_neg')
    
    # Creating a negative lst raster for filtering
    lst_neg = lst_dem.multiply(-1).rename('lst_neg')
    
    # Lst for non water pixels 
    lst_nw = lst_dem.updateMask(ndwi.lte(0)).rename('lst_nw')
 
    # Homogeneous mask for ndvi
    stdev_ndvi = homogeneous_mask(ndvi, proj)

    # Flat areas mask
    slope_filt = slope_mask(elev, proj) 

    # Creating a raster with all the parameters
    images = pos_ndvi.addBands([ndvi, ndvi_neg, rn, g, pos_ndvi, lst_neg, lst_nw, coords])
    
    # Estimating ndvi percentile [hottest]
    d_perc_down_ndvi = (
        images.select('post_ndvi')
        .updateMask(slope_filt)
        .updateMask(stdev_ndvi)
        .reduceRegion(
            reducer=ee.Reducer.percentile([ndvi_hot]),
            geometry=geometry_image,
            scale=30,
            maxPixels=1e9,
        )
        .combine(ee.Dictionary({'post_ndvi': 100}), overwrite=False)
    )

    # Get top ndvi value
    n_perc_low_NDVI = ee.Number(d_perc_down_ndvi.get('post_ndvi'))

    # Filtering ndvi raster with the percecilte
    i_low_NDVI = images.updateMask(images.select('post_ndvi').lte(n_perc_low_NDVI))

    # Estimating lst percentile [hottest]
    d_perc_top_lst = (
        i_low_NDVI.select('lst_neg')
        .reduceRegion(
            reducer=ee.Reducer.percentile([lst_hot]),
            geometry=geometry_image,
            scale=30,
            maxPixels=1e9,
        )
        .combine(ee.Dictionary({'lst_neg': 350}), overwrite=False)
    )
    
    # Get low lst value
    n_perc_top_lst = ee.Number(d_perc_top_lst.get('lst_neg'))

    # Filtering lst raster with the percecilte  
    i_top_LST = i_low_NDVI.updateMask(i_low_NDVI.select('lst_neg').lte(n_perc_top_lst))

    c_lst_hot_int = i_top_LST.select('lst_nw').min(1).max(1).int().rename('int')
    c_lst_hotpix = i_top_LST.addBands(c_lst_hot_int).addBands(tfac)

    # Getting the number of pixels found
    sum_final_hot_pix = c_lst_hotpix.select('int').reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry_image,
        scale=30,
        maxPixels=1e9,
    )

    n_sum_final_hot_pix = ee.Number(sum_final_hot_pix.get('int'))

    # Define Hot Pixel 
    # 5000 random pixels and select the 200 coldest
    def set_pt_geometry(f):
        return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))

    fc_hot_pix = (
        c_lst_hotpix.stratifiedSample(5000, 'int', geometry_image, 30)
        .map(set_pt_geometry)
        .sort('lst_nw', False)  
        .limit(200)
    )  

    # Aggregate all into lists
    list_long_hot = fc_hot_pix.aggregate_array('longitude')
    list_lat_hot = fc_hot_pix.aggregate_array('latitude')

    n_Ts_hot = ee.Number(fc_hot_pix.aggregate_mean('lst_nw'))
    n_long_hot = ee.Number(fc_hot_pix.aggregate_first('longitude'))
    n_lat_hot = ee.Number(fc_hot_pix.aggregate_first('latitude'))
    n_ndvi_hot = ee.Number(fc_hot_pix.aggregate_first('post_ndvi'))
    n_Rn_hot = ee.Number(fc_hot_pix.aggregate_mean('rn_inst'))
    n_G_hot = ee.Number(fc_hot_pix.aggregate_mean('g_inst'))
    n_Tfac = ee.Number(fc_hot_pix.aggregate_first('tfac'))

    d_hot_pixel = ee.Dictionary({
        'temp': n_Ts_hot,
        'tfac': n_Tfac,
        'x': n_long_hot,
        'y': n_lat_hot,
        'list_x':list_long_hot,
        'list_y':list_lat_hot,
        'Rn': n_Rn_hot,
        'G': n_G_hot,
        'ndvi': n_ndvi_hot,
        'sum': n_sum_final_hot_pix,
    })
 
    return d_hot_pixel.combine(
        ee.Dictionary({
        'temp': 0, 'tfac': 0, 'x': 0, 'y': 0, 'Rn': 0, 'G': 0, 
        'ndvi': 0, 'sum': 0, 'list_x':[0], 'list_y':[0]}), False
        )


def sensible_heat_flux(lst_dem, savi, ux, n_Ts_cold, d_hot_pixel, elev):
    """
    Instantaneous Sensible Heat Flux [W m-2]

    Parameters
    ----------
    lst_dem : ee.Image
        Land surface temperature [K].
    savi : ee.Image
        Soil-adjusted vegetation index.
    ux : ee.Image
        Wind speed [m s-1].
    n_Ts_cold : ee.FeatureCollection
        Cold pixels.
    d_hot_pixel : ee.FeatureCollection
        Hot pixels.
    elev : ee.Image
        Elevation [m].

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Bastiaanssen1998] Bastiaanssen, W.G.M., Menenti, M., Feddes, R.A.,
        Holtslag, A.A.M., 1998. A remote sensing surface energy balance
        algorithm for land (SEBAL): 1. Formulation. J. Hydrol. 212–213, 198–212.
    .. [Allen2002] Allen, R., Bastiaanssen., W.G.M. 2002.
        Surface Energy Balance Algorithms for Land. Idaho Implementation.
        Advanced Training and Users Manual. 2002.
    """

    # Vegetation height [m]
    n_veg_height = ee.Number(0.5)

    # Wind speed height [m]
    n_zx = ee.Number(2)

    # Blending height [m]
    n_height = ee.Number(200)

    # Air specific heat [J kg-1 K-1]
    n_Cp = ee.Number(1004)

    # Von Karman’s constant
    n_K = ee.Number(0.41)

    # Lst cold pixel value
    n_Ts_cold = ee.Number(n_Ts_cold)

    # Lst hot pixel value
    n_Ts_hot = ee.Number(d_hot_pixel.get('temp')).subtract(ee.Number(d_hot_pixel.get('tfac')))  ### MODIFICACAO

    # G inst hot pixel value
    n_G_hot = ee.Number(d_hot_pixel.get('G'))

    # Rn inst hot pixel value
    n_Rn_hot = ee.Number(d_hot_pixel.get('Rn'))
    
    # Hot pixels coordinates
    list_long_hot = ee.List(d_hot_pixel.get('list_x'))
    list_lat_hot = ee.List(d_hot_pixel.get('list_y'))

    # Momentum roughness length at the weather station (Allen2002 Eqn 28)
    n_zom = ee.Number(n_veg_height).multiply(0.12)

    # We use this to iterate through the indices of your coordinate lists
    indices = ee.List.sequence(0, list_long_hot.length().subtract(1))

    # Map over the indices to create a FeatureCollection of Points
    def create_points(i):
        return ee.Feature(ee.Geometry.Point([
            list_long_hot.get(i), 
            list_lat_hot.get(i)
        ]))

    pts_collection = ee.FeatureCollection(indices.map(create_points))

    # Friction velocity at the weather station (Allen2002 Eqn 37)
    i_ufric_ws = savi.expression(
        '(n_K * ux) / log(n_zx / n_zom)',
        {'n_K': n_K, 'n_zx': n_zx, 'n_zom': n_zom, 'ux': ux}
    )
    
    # Wind speed at blending height at the weather station  (Allen2002 Eqn 29)
    i_u200 = savi.expression(
        'i_ufric_ws * log(n_height / n_zom) / n_K',
        {'i_ufric_ws': i_ufric_ws, 'n_height': n_height, 'n_zom': n_zom, 'n_K': n_K}
    )
    
    # Momentum roughness length for each pixel
    i_zom = savi.expression('exp((5.62 * SAVI) - 5.809)', {'SAVI': savi})

    # Slope/ Aspect
    slope_aspect = ee.Terrain.products(elev)
    
    # Momentum roughness slope/aspect Correction  (Allen2002  A12 Eqn9)
    i_zom = i_zom.expression(
        'zom * (1 + (slope - 5) / 20)',
        {'zom': i_zom, 'slope': slope_aspect.select('slope')}
    )
    
    # Friction velocity for each pixel (Allen2002 Eqn 30)
    i_ufric = i_zom.expression(
        '(n_K * u200) / log(height / i_zom)',
        {'u200': i_u200, 'height': n_height, 'i_zom': n_zom, 'n_K': n_K}
    ).rename('u_fr')
    
    # Heights [m] above the zero plane displacement
    z1 = ee.Number(0.01)
    z2 = ee.Number(2)

    i_rah = i_ufric.expression( 
        '(log(z2 / z1)) / (i_ufric * 0.41)', {'z2': z2, 'z1': z1, 'i_ufric': i_ufric}
    ).rename('rah') 

    n_ro_hot = (ee.Number(-0.0046).multiply(n_Ts_hot)).add(ee.Number(2.5538))
    n_H_hot = ee.Number(n_Rn_hot).subtract(ee.Number(n_G_hot))
    
    # Initial value for iterative loop
    i_H_int = None
    i_ufric_loop = i_ufric # Copy to avoid overwritting the original base image

    # ITERATIVE PROCESS
    for n in range(1, 8):
        sampled_points = i_rah.select('rah').reduceRegions(
            collection=pts_collection,
            reducer=ee.Reducer.mean(), # Gets the pixel value at each point
            scale=30,
        )

        n_rah_hot = ee.Number(sampled_points.aggregate_mean('mean'))

        n_dT_hot = (n_H_hot.multiply(n_rah_hot)).divide(n_ro_hot.multiply(n_Cp))
        n_dT_cold = ee.Number(0)
        
        n_coef_a = (n_dT_cold.subtract(n_dT_hot)).divide(n_Ts_cold.subtract(n_Ts_hot))
        n_coef_b = n_dT_hot.subtract(n_coef_a.multiply(n_Ts_hot))

        i_dT_int = ee.Image(0).expression(
            '(n_coef_a * i_lst_med) + n_coef_b',
            {'n_coef_a': n_coef_a, 'n_coef_b': n_coef_b, 'i_lst_med': lst_dem}
        ).rename('dT')
    
        i_Ta = lst_dem.subtract(i_dT_int)
        i_ro = i_Ta.expression('(-0.0046 * i_Ta) + 2.5538', {'i_Ta': i_Ta}).rename('ro')
     
        i_H_step = i_dT_int.expression( 
            '(i_ro * n_Cp * i_dT_int) / i_rah',
            {'i_ro': i_ro, 'n_Cp': n_Cp, 'i_dT_int': i_dT_int, 'i_rah': i_rah}
        ).rename('H')
          
        if n == 1:
            i_H_int = i_H_step
        else:
            i_H_int = i_H_step.add(i_H_int).divide(2)
          
        i_L_int = i_dT_int.expression( 
            '-(i_ro * n_Cp * (i_ufric ** 3) * i_lst_med) / (0.41 * 9.81 * i_H_int)',
            {
                'i_ro': i_ro,
                'n_Cp': n_Cp,
                'i_ufric': i_ufric_loop,
                'i_lst_med': lst_dem,
                'i_H_int': i_H_int 
            }
        ).rename('L')
          
        img_zero = ee.Image(0)
        
        # Stability Corrections
        i_psim_200 = img_zero.expression('-5 * (2 / i_L_int)', {'i_L_int': i_L_int})
        i_psih_2 = img_zero.expression('-5 * (2 / i_L_int)', {'i_L_int': i_L_int})
        i_psih_01 = img_zero.expression('-5 * (0.1 / i_L_int)', {'i_L_int': i_L_int})
      
        i_x200 = i_L_int.expression('(1 - (16 * (200 / i_L_int))) ** 0.25', {'i_L_int': i_L_int})
        i_x2 = i_L_int.expression('(1 - (16 * (2 / i_L_int))) ** 0.25', {'i_L_int': i_L_int})
        i_x01 = i_L_int.expression('(1 - (16 * (0.1 / i_L_int))) ** 0.25', {'i_L_int': i_L_int})
      
        pi_val = 3.14159265
        i_psimu_200 = i_x200.expression(
            '2 * log((1 + x) / 2) + log((1 + x ** 2) / 2) - 2 * atan(x) + 0.5 * pi',
            {'x': i_x200, 'pi': pi_val}
        )
        i_psihu_2 = i_x2.expression('2 * log((1 + x ** 2) / 2)', {'x': i_x2})
        i_psihu_01 = i_x01.expression('2 * log((1 + x ** 2) / 2)', {'x': i_x01})
      
        i_psim_200 = i_psim_200.where(i_L_int.lt(0), i_psimu_200).where(i_L_int.eq(0), 0)
        i_psih_2 = i_psih_2.where(i_L_int.lt(0), i_psihu_2).where(i_L_int.eq(0), 0)
        i_psih_01 = i_psih_01.where(i_L_int.lt(0), i_psihu_01).where(i_L_int.eq(0), 0)
    
        i_ufric_step = i_ufric_loop.expression( 
            '(u200 * 0.41) / (log(hight / i_zom) - i_psim_200)',
            {'u200': i_u200, 'hight': n_height, 'i_zom': i_zom, 'i_psim_200': i_psim_200}
        )
        
        if n == 1:
            i_ufric_loop = i_ufric_step
        else:
            i_ufric_loop = i_ufric_step.add(i_ufric_loop).divide(2)
         
        i_rah = i_rah.expression( 
            '(log(z2 / z1) - psi_h2 + psi_h01) / (i_ufric * 0.41)',
            {'z2': z2, 'z1': z1, 'i_ufric': i_ufric_loop, 'psi_h2': i_psih_2, 'psi_h01': i_psih_01}
        ).rename('rah')

    # FINAL ESTIMATES
    i_rah_final = i_rah.clamp(0.01, 500).rename('rah')
    i_dT_final = i_dT_int.rename('dT')
    
    i_H_final = i_H_int.expression(
        '(i_ro * n_Cp * i_dT_int) / i_rah',
        {'i_ro': i_ro, 'n_Cp': n_Cp, 'i_dT_int': i_dT_final, 'i_rah': i_rah_final}
    ).clamp(-200, 600).rename('H')
      
    return i_H_final


def daily_et(h_inst, g_inst, rn_inst, lst_dem, rad_24h):
    """
    Daily Evapotranspiration [mm day-1]

    Parameters
    ----------
    h_inst : ee.Image
        Instantaneous Sensible heat flux [W m-2].
    g_inst : ee.Image
        Instantaneous Soil heat flux [W m-2].
    rn_inst : ee.Image
        Instantaneous Net radiation [ W m-2].
    lst_dem : ee.Image
        Land surface temperature (aspect/slope correction) [K].
    rad_24h : ee.Image
        Daily Net Radiation [W m-2].

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Bastiaanssen1998] Bastiaanssen, W.G.M., Menenti, M., Feddes, R.A.,
        Holtslag, A.A.M., 1998. A remote sensing surface energy balance
        algorithm for land (SEBAL): 1. Formulation. J. Hydrol. 212–213, 198–212.
    """

    # Instantaneous Latent Heat flux [W m-2]
    le_inst = h_inst.expression(
        '(i_Rn - i_G - i_H)', {'i_Rn': rn_inst, 'i_G': g_inst, 'i_H': h_inst}
    )

    # Latent heat of vaporization or the heat
    # absorbed when a kilogram of water evaporates [J/kg].
    i_lambda = h_inst.expression('(2.501 - 0.002361 * (Ts - 273.15))', {'Ts': lst_dem})

    # Evaporative fraction
    i_FE = h_inst.expression(
        'le_inst / (i_Rn - i_G)', {'le_inst': le_inst, 'i_Rn': rn_inst, 'i_G': g_inst}
    )

    # clamping between 0 and 1
    i_FE = i_FE.clamp(0, 1)

    # Calculating ET using Rn24h
    daily_et = i_FE.expression(
        '(0.0864 * i_FE * rad_24h) / i_lambda',
        {'i_FE': i_FE, 'i_lambda': i_lambda, 'rad_24h': rad_24h}
    )

    return daily_et.rename('et')


def et_fraction(time_start, et, et_reference_source, et_reference_band, et_reference_factor):
    """
    ET Fraction

    Parameters
    ----------
    time_start : ee.Image
        Instantaneous Sensible heat flux [W m-2].
    et : ee.Image
        Daily evapotranspiration (et) [mm day-1]
    et_reference_source : ee.ImageCollection, str
        ET reference collection
    et_reference_band : str
        ETr band name.
    et_reference_factor : ee.Number, int
        ETr factor.

    Returns
    -------
    ee.Image
    """

    date = ee.Date(time_start)
    start_date = ee.Date(utils.date_to_time_0utc(date))

    eto = (
        ee.ImageCollection(et_reference_source)
        .select(et_reference_band)
        .filterDate(start_date, start_date.advance(1, 'day'))
    )
    et_reference_img = ee.Image(eto.first())
    et_reference_img = et_reference_img.multiply(et_reference_factor)

    et_fraction = et.divide(et_reference_img).rename('et_fraction')

    return et_fraction
