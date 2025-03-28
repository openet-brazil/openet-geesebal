import math

import ee

# from openet.geesebal import openet_landsat as landsat
from openet.geesebal import utils

DEG2RAD = math.pi / 180.0


def et(
    image,
    ndvi,
    ndwi,
    lst,
    albedo,
    emissivity,
    savi,
    meteorology_source_inst,
    meteorology_source_daily,
    elev_product,
    ndvi_cold,
    ndvi_hot,
    lst_cold,
    lst_hot,
    time_start,
    geometry_image,
    proj,
    coords,
    #et_reference,
    cold_calibration_points=10,
    hot_calibration_points=10,
    max_iterations=15,
):
    """
    Daily Evapotranspiration [mm day-1].

    Parameters
    ----------
    image : ee.Image
        Landsat image.
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
    meteorology_source_inst : ee.ImageCollection
        Meteorological dataset [inst]
    meteorology_source_daily : ee.ImageCollection
        Meteorological dataset [daily]
    elev_product : ee.Image
    ndvi_cold : ee.Number, int
        NDVI Percentile value to determinate cold pixel.
    ndvi_hot : ee.Number, int
        NDVI Percentile value to determinate hot pixel.
    lst_cold : ee.Number, int
        Lst Percentile value to determinate cold pixel.
    lst_hot : ee.Number, int
        Lst Percentile value to determinate hot pixel.
    time_start : str
        Image property: time start of the image.
    geometry_image : ee.Geometry
        Image geometry.
    proj : ee.Image
        Landsat image projection.
    coords : ee.Image
        Landsat image Latitude and longitude.
    cold_calibration_points : int
        Number of cold pixel calibration points (the default is 10).
    hot_calibration_points : int
        Number of hot pixel calibration points (the default is 10).
    max_iterations : int
        Maximum number of iterations (the default is 15).

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
    hour = ee.Number(date.get("hour"))
    minutes = ee.Number(date.get("minutes"))

    # Endmembers
    top_ndvi = ee.Number(ndvi_cold)
    coldest_lst = ee.Number(lst_cold)
    lowest_ndvi = ee.Number(ndvi_hot)
    hottest_lst = ee.Number(lst_hot)

    # Meteo source
    # TODO: check how to add CIMIS data
    if (meteorology_source_inst == "NASA/NLDAS/FORA0125_H002") and \
                    (meteorology_source_daily == "IDAHO_EPSCOR/GRIDMET"):
        tmin, tmax, tair, ux, rh, rso_inst, rso24h, tfac = meteorology_nldas_gridmet(
            time_start,
            meteorology_source_inst,
            meteorology_source_daily,
        )

    elif (meteorology_source_inst == "ECMWF/ERA5_LAND/HOURLY") and \
            (meteorology_source_daily == "projects/openet/assets/meteorology/era5land/na/daily") or \
                (meteorology_source_inst == "ECMWF/ERA5_LAND/HOURLY") and \
                    (meteorology_source_daily == "projects/openet/assets/meteorology/era5land/sa/daily"):

        tmin, tmax, tair, ux, rh, rso_inst, rso24h, tfac = meteorology_era5land(
            time_start,
            meteorology_source_inst,
            meteorology_source_daily,
        )

    else:
        raise Exception("Error: wrong daily or instant met data source assigned.")

    # Elevation data [m]
    dem_product = ee.Image(elev_product)
    elev = dem_product.select("elevation")

    # Sun elevation []
    sun_elevation = ee.Number(image.get("SUN_ELEVATION"))

    # Terrain cos
    cos_zn = cos_terrain(time_start, dem_product, hour, minutes, coords)

    # LL: iterative process or endmembers selections may return empty values
    # in this case, return an empty image instead of broken the code
    try:
            # Air temperature correction [K]
        tair_dem = tair_dem_correction(tmin, tmax, elev)

        # Land surface temperature correction [K]
        lst_dem = lst_correction(time_start, lst, elev, tair_dem, rh, sun_elevation,
                                                    hour, minutes, coords, geometry_image)

        # Instantaneous net radiation using reanalysis dataset [W m-2]
        rad_inst = radiation_inst(elev, lst, emissivity, albedo, tair, rh, rso_inst, sun_elevation, cos_zn)

        # Instantaneous soil heat flux [W m-2]
        g_inst = soil_heat_flux(rad_inst, ndvi, albedo, lst_dem, ndwi)

        # Daily ney radiation [W m-2]
        rad_24h = radiation_24h(time_start, tmax, tmin, elev, sun_elevation, cos_zn, rso24h)

        # Cold pixel for wet conditions repretation of the image
        cold_pixels = cold_pixel(albedo, ndvi, ndwi, lst_dem, top_ndvi, coldest_lst,
                                        geometry_image, coords, proj, elev, cold_calibration_points)
        # Hot pixel
        hot_pixels = fexp_hot_pixel(time_start, albedo, ndvi, ndwi, lst,lst_dem, rad_inst,
                                        g_inst, tair, ux, lowest_ndvi, hottest_lst, geometry_image,
                                            coords, proj, elev, tfac, hot_calibration_points)
        # Instantaneous sensible heat flux [W m-2]
        h_inst = sensible_heat_flux(savi, ux, cold_pixels, hot_pixels, lst_dem, lst,
                                        elev, geometry_image, max_iterations)
        
        # Checking if H was estimated, otherwise return a nodata mask
        h_cond = ee.Number(cold_pixels.size()).eq(0).Or(ee.Number(hot_pixels.size()).eq(0))

        h_inst = ee.Image(ee.Algorithms.If(h_cond.eq(1),
                                        ee.Image.constant(0).updateMask(0),
                                        h_inst)).rename("h_inst")

        # Daily evapotranspiration [mm day-1]
        et_24hr = daily_et(h_inst, g_inst, rad_inst, lst_dem, rad_24h)

    except Exception as e:
        # CGM - We should probably log the exception so the user knows,
        #   but this will cause problems when mapping over a collection
        print(f"Unhandled Exception: {e}")

        # Return a masked image
        et_24hr = ee.Image.constant(0).updateMask(0).rename("et")

    return et_24hr.rename("et")


def meteorology_nldas_gridmet(time_start, meteorology_source_inst, meteorology_source_daily):
    """
    Parameters
    ----------
    time_start : str
        Image property: time start of the image.
    meteorology_source_inst: ee.ImageCollection, str
        Instantaneous meteorological data.
    meteorology_source_daily :  ee.ImageCollection, str
        Daily meteorological data.

    Returns
    -------
    ee.Image

    Notes
    -----
    Accepted collections:
    Inst : NASA/NLDAS/FORA0125_H002
    Daily : IDAHO_EPSCOR/GRIDMET

    References
    ----------

    """
    # Get date information
    time_start = ee.Number(time_start)

    # Filtering Daily data
    meteorology_daily = ee.ImageCollection(meteorology_source_daily).filterDate(
        ee.Date(time_start).advance(-1, "day"), ee.Date(time_start)
    )

    # Instantaneous data
    meteorology_inst_collection = ee.ImageCollection(meteorology_source_inst)

    # Linear interpolation
    previous_time = time_start.subtract(2 * 60 * 60 * 1000)
    next_time = time_start.add(2 * 60 * 60 * 1000)

    previous_image = (
        meteorology_inst_collection.filterDate(previous_time, time_start).limit(1, "system:time_start", False).first()
    )

    next_image = (
        meteorology_inst_collection.filterDate(time_start, next_time).limit(1, "system:time_start", True).first()
    )

    image_previous_time = ee.Number(previous_image.get("system:time_start"))
    image_next_time = ee.Number(next_image.get("system:time_start"))
    delta_time = time_start.subtract(image_previous_time).divide(image_next_time.subtract(image_previous_time))

    # Incoming shorwave down [W m-2]
    swdown24h = meteorology_daily.select("srad").first().rename("short_wave_down")

    # Minimum air tempreature [K]
    tmin = meteorology_daily.select("tmmn").first().rename("tmin")

    # Maximum air temperature [K]
    tmax = meteorology_daily.select("tmmx").first().rename("tmax")

    # Instantaneous short wave radiation [W m-2]
    rso_inst = (
        next_image.select("shortwave_radiation")
        .subtract(previous_image.select("shortwave_radiation"))
        .multiply(delta_time)
        .add(previous_image.select("shortwave_radiation"))
        .rename("rso_inst")
    )

    # Specific humidity [Kg Kg-1]
    q_med = (
        next_image.select("specific_humidity")
        .subtract(previous_image.select("specific_humidity"))
        .multiply(delta_time)
        .add(previous_image.select("specific_humidity"))
    )

    # Air temperature [K]
    tair_c = (
        next_image.select("temperature")
        .subtract(previous_image.select("temperature"))
        .multiply(delta_time)
        .add(previous_image.select("temperature"))
        .rename("tair")
    )

    # Wind speed u [m s-1]
    wind_u = (
        next_image.select("wind_u")
        .subtract(previous_image.select("wind_u"))
        .multiply(delta_time)
        .add(previous_image.select("wind_u"))
    )

    # Wind speed u [m s-1]
    wind_v = (
        next_image.select("wind_v")
        .subtract(previous_image.select("wind_v"))
        .multiply(delta_time)
        .add(previous_image.select("wind_v"))
    )

    wind_med = wind_u.expression("sqrt(ux_u ** 2 + ux_v ** 2)", {"ux_u": wind_u, "ux_v": wind_v}).rename("ux")

    # Wind speed [m s-1] (FAO56 Eqn 47)
    wind_med = wind_med.expression("ux * (4.87) / log(67.8 * z - 5.42)", {"ux": wind_med, "z": 10.0})

    # Pressure [kPa]
    p_med = (
        next_image.select("pressure")
        .subtract(previous_image.select("pressure"))
        .multiply(delta_time)
        .add(previous_image.select("pressure"))
        .divide(ee.Number(1000))
    )

    # Actual vapor pressure [kPa] (Shuttleworth Eqn 2.10)
    ea = p_med.expression("(1 / 0.622) * Q * P", {"Q": q_med, "P": p_med})

    # Saturated vapor pressure [kPa] (FAO56 Eqn 11)
    esat = tair_c.expression("0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))", {"T_air": tair_c})

    # Relative humidity (%)  (FAO56 Eqn 10)
    rh = ea.divide(esat).multiply(100).rename("RH")

    # Surface temperature correction based on precipitation and reference ET

    # Accumulation time period
    accum_period = -60

    # Accum meteo data 
    gridmet_accum = ee.ImageCollection(meteorology_source_daily).filterDate(
        ee.Date(time_start).advance(accum_period, "days"), ee.Date(time_start)
    )

    # Reference ET 
    etr_accum = gridmet_accum.select("etr").sum()

    # Precipitation
    precipt_accum = gridmet_accum.select("pr").sum()

    # Ratio between precipt/etr
    ratio = precipt_accum.divide(etr_accum)

    # Temperature adjustment offset (Allen2013 Eqn 8)
    
    tfac = etr_accum.expression("2.6 - 13 * ratio", {"ratio": ratio})

    tfac = ee.Image(tfac.where(ratio.gt(0.2), 0)).rename("tfac")

    # Resample
    tmin = tmin.subtract(273.15).resample("bilinear")
    tmax = tmax.subtract(273.15).resample("bilinear")
    rso_inst = rso_inst.resample("bilinear")
    tair_c = tair_c.resample("bilinear")
    wind_med = wind_med.resample("bilinear")
    rh = rh.resample("bilinear")
    swdown24h = swdown24h.resample("bilinear")

    return [tmin, tmax, tair_c, wind_med, rh, rso_inst, swdown24h, tfac]


def meteorology_era5land(time_start, meteorology_source_inst, meteorology_source_daily):
    """
    Parameters
    ----------
    time_start : str
        Image property: time start of the image.
    meteorology_source_inst: ee.ImageCollection, str
        Instantaneous meteorological data.
    meteorology_source_daily :  ee.ImageCollection, str
        Daily meteorological data.

    Returns
    -------
    ee.Image

    Notes
    -----
    Accepted collections:
    Inst : ECMWF/ERA5_LAND/HOURLY
    Daily : projects/openet/assets/meteorology/era5land/na/daily
            projects/openet/assets/meteorology/era5land/sa/daily

    References
    ----------

    """

    # Get date information
    time_start = ee.Number(time_start)

    # Filtering Daily data
    meteorology_daily = (
        ee.ImageCollection(meteorology_source_daily)
        .filterDate(ee.Date(time_start).advance(-1, "day"), ee.Date(time_start).advance(1, "day"))
        .first()
    )

    # Instantaneous data
    meteorology_inst_collection = ee.ImageCollection(meteorology_source_inst)

    # Linear interpolation
    previous_time = time_start.subtract(1 * 60 * 60 * 1000)
    next_time = time_start.add(1 * 60 * 60 * 1000)

    previous_image = (
        meteorology_inst_collection.filterDate(previous_time, time_start).limit(1, "system:time_start", False).first()
    )

    next_image = (
        meteorology_inst_collection.filterDate(time_start, next_time).limit(1, "system:time_start", True).first()
    )

    image_previous_time = ee.Number(previous_image.get("system:time_start"))
    image_next_time = ee.Number(next_image.get("system:time_start"))

    delta_time = time_start.subtract(image_previous_time).divide(image_next_time.subtract(image_previous_time))

    # Incoming shorwave down [W m-2]
    swdown24h = meteorology_daily.select("surface_solar_radiation_downwards").divide(1 * 60 * 60 * 24)

    # Minimum air temperature [K]
    tmin = meteorology_daily.select("temperature_2m_min").rename("tmin")
    
    # Maximum air temperature [K]
    tmax = meteorology_daily.select("temperature_2m_max").rename("tmax")

    # Instantaneous incoming shortwave radiation [W m-2]
    rso_inst = (
        ee.ImageCollection(meteorology_source_inst)
        .filterDate(ee.Date(time_start), ee.Date(time_start).advance(1, "hour"))
        .select("surface_solar_radiation_downwards_hourly")
        .mean()
        .divide(1 * 60 * 60)
        .rename("rso_inst")
    )

    # Air temperature [C]
    # TODO: LL- Change all temperatures to K ?
    tair_c = (
        next_image.select("temperature_2m")
        .subtract(previous_image.select("temperature_2m"))
        .multiply(delta_time)
        .add(previous_image.select("temperature_2m"))
        .subtract(273.15)
        .rename("tair")
    )

    # Wind speed [ m/s]
    wind_u = (
        next_image.select("u_component_of_wind_10m")
        .subtract(previous_image.select("u_component_of_wind_10m"))
        .multiply(delta_time)
        .add(previous_image.select("u_component_of_wind_10m"))
    )

    wind_v = (
        next_image.select("v_component_of_wind_10m")
        .subtract(previous_image.select("v_component_of_wind_10m"))
        .multiply(delta_time)
        .add(previous_image.select("v_component_of_wind_10m"))
    )

    wind_med = wind_u.expression(
        "sqrt(ux_u ** 2 + ux_v ** 2)",
        {"ux_u": wind_u, "ux_v": wind_v},
    ).rename("ux")

    wind_med = wind_med.expression("ux * (4.87) / log(67.8 * z - 5.42)", {"ux": wind_med, "z": 10.0}).rename("ux")

    # Dew point temperature [°K]
    tdp = (
        next_image.select("dewpoint_temperature_2m")
        .subtract(previous_image.select("dewpoint_temperature_2m"))
        .multiply(delta_time)
        .add(previous_image.select("dewpoint_temperature_2m"))
        .rename("tdp")
    )

    # Actual vapour pressure [kPa]
    ea = tdp.expression("0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))", {"T_air": tdp.subtract(273.15)})

    # SATURATED VAPOR PRESSURE [kPa]
    esat = tair_c.expression("0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))", {"T_air": tair_c})

    # RELATIVE HUMIDITY (%)
    rh = ea.divide(esat).multiply(100).rename("RH")

    # Surface temperature correction based on precipitation and reference ET

    # Accumulation time period
    accum_period = -60

    # Accum meteo data 
    gridmet_accum = ee.ImageCollection(meteorology_source_daily).filterDate(
        ee.Date(time_start).advance(accum_period, "days"), ee.Date(time_start)
    )

    # Reference ET 
    etr_accum = gridmet_accum.select("etr_asce").sum()

    # Precipitation
    precipt_accum = gridmet_accum.select("total_precipitation").sum()

    # Ratio between precipt/etr
    ratio = precipt_accum.divide(etr_accum)

    # Temperature adjustment offset (Allen2013 Eqn 8)
    
    tfac = etr_accum.expression("2.6 - 13 * ratio", {"ratio": ratio})

    tfac = ee.Image(tfac.where(ratio.gt(0.2), 0)).rename("tfac")

    # Resample
    tmin = tmin.subtract(273.15).resample("bilinear")
    tmax = tmax.subtract(273.15).resample("bilinear")
    rso_inst = rso_inst.resample("bilinear")
    tair_c = tair_c.resample("bilinear")
    wind_med = wind_med.resample("bilinear")
    rh = rh.resample("bilinear")
    swdown24h = swdown24h.resample("bilinear")

    return [tmin, tmax, tair_c, wind_med, rh, rso_inst, swdown24h, tfac]


def tao_sw(dem, tair, rh, sun_elevation, cos_zn):
    """
    Correct declivity and aspect effects from Land Surface Temperature.

    Parameters
    ----------
    dem : ee.Image
        Elevation product data [m].
    tair : ee.Image
        Air temperature [Celsius].
    rh : ee.Image
        Relative Humidity [%]
    sun_elevation : ee.Number, ee.Image
        Sun elevation angle.
    cos_zn : ee.Number, ee.Image
        Solar zenith angle cos.

    Returns
    -------
    ee.Image

    References
    ----------
    """
    # Atmospheric pressure [kPa] (FAO56 Eqn 7)
    pres = dem.expression("101.3 * ((293 - (0.0065 * Z)) / 293) ** 5.26 ", {"Z": dem})

    # Saturated vapor pressure [kPa] (FAO56 Eqn 11)
    es = tair.expression("0.6108 * exp((17.27 * tair) / (tair + 237.3))", {"tair": tair})

    # Actual vapor pressure [kPa]  (FAO56 Eqn 10)
    ea = es.multiply(rh).divide(100).rename("ea")

    # Water in the atmosphere [mm] (Garrison and Adler (1990))
    w = ea.expression("(0.14 * EA * PATM) + 2.1", {"PATM": pres, "EA": ea})

    # Solar angle
    sin_zn = sun_elevation.multiply(DEG2RAD).sin()

    # Solar zenith angle over a horizontal surface
    #solar_zenith = ee.Number(90).subtract(sun_elevation)

    #solar_zenith_radians = solar_zenith.multiply(DEG2RAD)
    # Cos only in flat areas
    # cos_theta = solar_zenith_radians.cos()

    # Broad-band atmospheric transmissivity (ASCE-EWRI (2005))
    #tao_sw_img = pres.expression(
    #    "0.35 + 0.627 * exp(((-0.00146 * P) / (Kt * ct)) - (0.075 * (W / ct) ** 0.4))",
    #    {"P": pres, "W": w, "Kt": 1.0, "ct": cos_zn},
    #)
    tao_sw_img = pres.expression(
        "0.35 + 0.627 * exp(((-0.00146 * P) / (Kt * ct)) - (0.075 * (W / ct) ** 0.4))",
        {"P": pres, "W": w, "Kt": 1.0, "ct": sin_zn},
    )

    return tao_sw_img.rename("tao_sw")


def cos_terrain(time_start, dem, hour, minutes, coords):
    """
    Cosine zenith angle elevation (Allen et al. (2006)).

    Parameters
    ----------
    time_start : str
        Image property: time start of the image.
    dem : ee.Image
        Elevation product data [m].
    hour : ee.Number, int
        Hour.
    minutes : ee.Number, int
        Minutes.
    coords : ee.Image
        Latitude and longitude of the image.

    Returns
    -------
    ee.Image

    References
    ----------
    """
    # Day of the year
    doy = ee.Date(time_start).getRelative("day", "year").add(1)

    # Slope and aspect
    slope_aspect = ee.Terrain.products(dem)

    # Variables
    B = doy.subtract(81).multiply(360 / 365)
    delta = ee.Number(23.45 * DEG2RAD).sin().multiply(B.multiply(DEG2RAD).sin()).asin()
    s = slope_aspect.select("slope").multiply(DEG2RAD)
    gamma = slope_aspect.select("aspect").subtract(180).multiply(DEG2RAD)
    phi = coords.select("latitude").multiply(DEG2RAD)

    # Constants
    delta = ee.Image(delta)
    a = (
        delta.sin()
        .multiply(phi.cos())
        .multiply(s.sin())
        .multiply(gamma.cos())
        .subtract(delta.sin().multiply(phi.sin().multiply(s.cos())))
    )
    b = (
        delta.cos()
        .multiply(phi.cos())
        .multiply(s.cos())
        .add(delta.cos().multiply(phi.sin().multiply(s.sin()).multiply(gamma.cos())))
    )
    c = delta.cos().multiply(s.sin()).multiply(gamma.sin())

    # Centroid image
    longitude_center = coords.select("longitude")

    delta_gtm = longitude_center.divide(ee.Image(15)).int()

    # Local hour time
    lht = delta_gtm.add(hour).add(minutes.divide(60))

    w = lht.subtract(12).multiply(15).multiply(DEG2RAD)

    # Cosine  zenith angle elevation
    cos_zn = w.expression("-a + b * w_cos + c * w_sin", {"a": a, "b": b, "c": c, "w_cos": w.cos(), "w_sin": w.sin()})

    return cos_zn


def tair_dem_correction(tmin, tmax, dem):
    """
    Correct Air temperature for mountain areas

    Parameters
    ----------
    tmin : ee.Image
        Minimum temperature [Celsius].
    tmax : ee.Image
        Maximum temperature [Celsius].
    dem : ee.Image
        Elevation product data [m].

    Returns
    -------
    ee.Image

    References
    ---------
    """
    tmin_dem = tmin.expression(
        "temp - 0.0065 * (dem - alt_meteo)", {"temp": tmin, "dem": dem.select("elevation"), "alt_meteo": ee.Number(2)}
    )
    tmax_dem = tmax.expression(
        "temp - 0.0065 * (dem - alt_meteo)", {"temp": tmax, "dem": dem.select("elevation"), "alt_meteo": ee.Number(2)}
    )

    tair_dem = tmin_dem.add(tmax_dem).divide(2)

    return tair_dem.rename("tair_dem")


def lst_correction(time_start, lst, dem, tair, rh, sun_elevation, hour, minutes, coords, geometry_image):
    """
    Correct declivity and aspect effects from Land Surface Temperature.

    Parameters
    ----------
    time_start : str
        Image property: time start of the image.
    lst : ee.Image
        Land surface temperature [K]
    dem : ee.Image
        Elevation product data [m].
    tair : ee.Image
        Air temperature [Celsius].
    rh : ee.Image
        Relative Humidity [%].
    sun_elevation : ee.Number, ee.Image
        Sun elevation angle.
    hour : ee.Number, int
        Hour.
    minutes : ee.Number, int
        Minutes.
    coords : ee.Image
        Landsat image Latitude and longitude.

    Returns
    -------
    ee.Image

    References
    ----------
    TODO: Add full Zaafar and Farah 2020 reference here

    """
    # Solar constant [W m-2]
    gsc = ee.Image.constant(1367)

    # Day of the year
    doy = ee.Date(time_start).getRelative("day", "year").add(1)

    # Inverse relative distance earth-sun (FAO56 Eqn 23)
    dr = doy.multiply(2 * math.pi / 365).cos().multiply(0.033).add(1)

    # Atmospheric pressure [kPa] (FAO56 Eqn 7)
    pres = lst.expression("101.3 * ((293 - (0.0065 * Z)) / 293) ** 5.26 ", {"Z": dem})

    # Solar zenith angle over a horizontal surface
    solar_zenith = ee.Number(90).subtract(sun_elevation)

    solar_zenith_radians = solar_zenith.multiply(DEG2RAD)
    cos_theta = solar_zenith_radians.cos()

    # Air density [Kg m-3]
    air_dens = lst.expression("(1000 * Pair) / (1.01 * LST * 287)", {"Pair": pres, "LST": lst})

    # Temperature lapse rate [K m-1]
    temp_lapse_rate = ee.Number(0.0065)

    # Correcting land surface temperature [K]
    temp_corr = lst.add(dem.select("elevation").multiply(temp_lapse_rate))

    cos_zn = cos_terrain(time_start, dem, hour, minutes, coords)

    # Broad-band atmospheric transmissivity (ASCE-EWRI (2005))
    tao_sw_img = tao_sw(dem, tair, rh, sun_elevation, cos_zn)

    # Corrected Land Surface temperature [K] (Zaafar and Farah (2020) Eqn 2)
    lst_dem = lst.expression(
        "Temp_corr + first_term * (cos_zn - cos_zenith_flat) / (air_dens * 1004 * 0.05)",
        {
            "Temp_corr": temp_corr,
            "first_term": gsc.multiply(dr).multiply(tao_sw_img),
            "cos_zenith_flat": cos_theta,
            "cos_zn": cos_zn,
            "air_dens": air_dens,
        },
    )

    return lst_dem.rename("lst_dem")

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

    References
    ----------

    """

    # Calculate NDVI standard deviation in pixel neighborhood
    sd_ndvi = (
        ndvi.reduceNeighborhood(
            reducer=ee.Reducer.stdDev(), kernel=ee.Kernel.square(radius=3, units="pixels"), skipMasked=False
        )
        .reproject(proj)
        .updateMask(1)
    )
    
    # Calculate mean NDVI in pixel neighborhood
    mean_ndvi = (
            ndvi.reduceNeighborhood(
                reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(radius=3, units="pixels"), skipMasked=False
            )
            .reproject(proj)
            .updateMask(1)
        )

    cv_mask = sd_ndvi.divide(mean_ndvi).lte(0.15).selfMask()

    return ee.Image(cv_mask)

def cold_pixel(
    albedo,
    ndvi,
    ndwi,
    lst_dem,
    ndvi_cold,
    lst_cold,
    geometry_image,
    coords,
    proj,
    dem,
    calibration_points=10,
):
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
    proj : ee.Dictionary
        Landsat image projection.
    dem : ee.Image
        Elevation data [m].
    calibration_points : int
        Number of calibration points (the default is 10).

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
    pos_ndvi = ndvi.updateMask(ndvi.gte(0)).rename("post_ndvi")

    # Creating a negative ndvi raster for filtering
    ndvi_neg = pos_ndvi.multiply(-1).rename("ndvi_neg")

    # Creating a negative lst raster for filtering
    lst_neg = lst_dem.multiply(-1).rename("lst_neg")

    # Lst for non water pixels 
    # lst_nw = lst_dem.updateMask(ndwi.lte(0)).rename('lst_nw')
    lst_nw = lst_dem.rename("lst_nw")

    # Homogenetou mask for ndvi
    stdev_ndvi = homogeneous_mask(ndvi, proj)

    # Creating a raster with all the parameters
    images = pos_ndvi.addBands([ndvi, ndvi_neg, pos_ndvi, lst_neg, lst_nw, coords, dem.toFloat()])

    # Estimating ndvi percentile [coldest]
    perc_top_ndvi = (images.select("ndvi_neg")
        .updateMask(stdev_ndvi)
        .reduceRegion(reducer=ee.Reducer.percentile([ndvi_cold]), geometry=geometry_image, scale=30, maxPixels=1e9)
        .combine(ee.Dictionary({"ndvi_neg": 100}), overwrite=False)
    )

    # Get top ndvi value
    perc_top_ndvi_value = ee.Number(perc_top_ndvi.get("ndvi_neg"))

    # Filtering ndvi raster with the percecilte
    top_ndvi = images.updateMask(stdev_ndvi)\
                        .updateMask(images.select("ndvi_neg").lte(perc_top_ndvi_value))
    
    # Filtering lst
    perc_low_lst = (
        top_ndvi.select("lst_nw")
        .updateMask(stdev_ndvi)
        .reduceRegion(reducer=ee.Reducer.percentile([lst_cold]), geometry=geometry_image, scale=30, maxPixels=1e9)
        .combine(ee.Dictionary({"lst_nw": 350}), overwrite=False)
    )

    # Get low lst value
    perc_low_lst_value = ee.Number(perc_low_lst.get("lst_nw"))

    # Filtering lst raster with the percentile
    coldest_lst = top_ndvi.updateMask(top_ndvi.select("lst_nw").lte(perc_low_lst_value))

    # Creating a mask
    masks = coldest_lst.select("lst_nw").mask().selfMask()

    # Masking inputs
    coldest_lst = images.updateMask(masks)

    # Removing lst bias values (low than 200K)
    lst_cold_perc = coldest_lst.updateMask(coldest_lst.select("lst_nw").gte(200))

    # Creating a int raster
    lst_cold_perc_int = masks.int().rename("int")
    # lst_cold_perc_int = lst_cold_perc.select('lst_nw').min(1).max(1).int().rename('int')

    lst_coldest_pixels = lst_cold_perc.addBands(lst_cold_perc_int).select(
        "ndvi", "lst_nw", "longitude", "latitude", "elevation", "int"
    )

    # Getting the number of pixels found
    sum_final_cold_pix = lst_coldest_pixels.select("int").reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry_image, scale=30, maxPixels=1e9
    )

    sum_final_cold_pix_value = ee.Number(sum_final_cold_pix.get("int"))

    # CGM - Not used anymore
    # def function_def_pixel(f):
    #     return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))

    # Creating a table with the cold pixels identified in the processed
    cold_pixels_table = lst_coldest_pixels.stratifiedSample(
                numPoints=calibration_points,
                classBand="int",
                region=geometry_image,
                scale=30,
                dropNulls=True,
                geometries=True,
            )

    # Checkin if there are at least 1 pixel found as cold pixel
    minimum_cold_pixels = 1 #3000

    cold_pixels_table = ee.FeatureCollection(ee.Algorithms.If(
            sum_final_cold_pix_value.gte(minimum_cold_pixels), 
            cold_pixels_table, 
            ee.FeatureCollection([ee.Feature(ee.Geometry.Point([0, 0]),
                    {'ndvi': 0,'lst_nw': 0,'longitude': 0,
                     'latitude': 0, 'elevation': 0,'int': 1})])
                     )
    )
    
    return cold_pixels_table

def radiation_inst(dem, lst, emissivity, albedo, tair, rh, swdown_inst, sun_elevation, cos_terrain):
    """
    Instantaneous Net Radiation [W m-2]

    Parameters
    ----------
    dem : ee.Image
        Digital elevation product [m].
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
    swdown_inst : ee.Image
        Instantaneous Short Wave radiation [W m-2].
    sun_elevation : ee.Number, int
        Sun elevation information.
    cos_terrain : ee.Image
        Solar zenith angle cos (aspect/slope).

    Returns
    -------
    ee.Image

    References
    ----------

    """

    # Up long wave radiation [W m-2]
    rad_long_up = lst.expression("emi * 5.67e-8 * (lst ** 4)", {"emi": emissivity, "lst": lst})

    # Transmissivity 
    tao_sw_img = tao_sw(dem, tair, rh, sun_elevation, cos_terrain)

    log_taosw = tao_sw_img.log()

    # Down long wave radiation [W m-2] 
    rad_long_down = lst.expression(
        "(0.85 * (- log_taosw) ** 0.09) * 5.67e-8 * (n_Ts_cold ** 4)",
        {"log_taosw": log_taosw, "n_Ts_cold": tair.add(273.15)},
    )

    # Solar zenith
    solar_zenith = ee.Number(90).subtract(sun_elevation)
    solar_zenith_radians = solar_zenith.multiply(DEG2RAD)
    cos_zeni = solar_zenith_radians.cos()

    # Obtaining a Short down wave radiation correcting mountainous effect 
    swdown_inst_dem = swdown_inst.multiply(cos_terrain.divide(cos_zeni))

    # Net radiation [W m-2]
    rn_inst = lst.expression(
        "((1 - alfa) * Rs_down) + Rl_down - Rl_up - ((1 - e_0) * Rl_down)",
        {
            "alfa": albedo,
            "Rs_down": swdown_inst_dem,
            "Rl_down": rad_long_down,
            "Rl_up": rad_long_up,
            "e_0": emissivity,
        },
    )

    return rn_inst.rename("rn_inst")


def soil_heat_flux(rn, ndvi, albedo, lst_dem, ndwi):
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
    lst_dem : ee.Image
        Land surface temperature [K].
    ndwi : ee.Image
        Normalized difference water index.

    Returns
    -------
    ee.Image

    References
    ----------

    """

    # Soil heat flux [W m-2]
    g = rn.expression(
        "rn * (lst - 273.15) * (0.0038 + (0.0074 * albedo)) * (1 - 0.98 * (ndvi ** 4))",
        {"rn": rn, "ndvi": ndvi, "albedo": albedo, "lst": lst_dem},
    )

    # Considering G as 50% of net radiation for water bodies
    g = g.where(ndwi.gt(0), rn.multiply(0.5))

    return g.rename("g_inst")


def radiation_24h(time_start, tmax, tmin, elev, sun_elevation, cos_terrain, rso24h):
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
    sun_elevation : ee.Number, int
        Sun elevation information.
    cos_terrain : ee.Image
        Solar zenith angle cos (aspect/slope).
    rso24h : ee.Image
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
    rs = rso24h.multiply(0.0864).rename("Rs")

    # Solar constant [MJ m-2]
    gsc = 0.0820

    # Day of the year
    doy = ee.Date(time_start).getRelative("day", "year").add(1)

    # Inverse relative distance earth-sun (FAO56 Eqn 23)
    dr = tmax.expression("1 + (0.033 * cos((2 * pi / 365) * doy))", {"doy": doy, "pi": math.pi})

    # Solar declination [rad] (FAO56 Eqn 24)
    sd = tmax.expression("0.40928 * sin(((2 * pi / 365) * doy) - 1.39)", {"doy": doy, "pi": math.pi})

    # Latitude of the image
    lat = tmax.pixelLonLat().select(["latitude"]).multiply(DEG2RAD).rename("latitude")

    #  Sunset hour angle [rad] (FAO56 Eqn 25)
    ws = tmax.expression("acos(-tan(Lat) * tan(Sd))", {"Lat": lat, "Sd": sd})

    # Extraterrestrial radiation [MJ m-2 d-1] (FAO56 Eqn 21)
    rad_a = tmax.expression("Ws * sin(Lat) * sin(Sd) + cos(Lat) * cos(Sd) * sin(Ws)", {"Ws": ws, "Lat": lat, "Sd": sd})

    ra = tmax.expression("((24 * 60) / pi) * Gsc * Dr * rad_a", {"pi": math.pi, "Gsc": gsc, "Dr": dr, "rad_a": rad_a})
    # Simplified clear sky solar formulation [MJ m-2 d-1] (FAO56 Eqn 37)
    rso = tmax.expression("(0.75 + 2E-5 * z) * Ra", {"z": elev, "Ra": ra})

    # Net shortwave radiation [MJ m-2 d-1] (FAO56 Eqn 38)
    rns = tmax.expression("(1 - albedo) * Rs", {"Rs": rs, "albedo": 0.23})

    # Actual vapor pressure [MJ m-2 d-1] (FAO56 Eqn 11)
    ea = tmax.expression("0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))", {"T_air": tmin})

    # Rso slope/aspect
    solar_zenith = ee.Number(90).subtract(sun_elevation)
    solar_zenith_radians = solar_zenith.multiply(DEG2RAD)
    cos_zeni = solar_zenith_radians.cos()

    rso24h_dem = rso.multiply(cos_terrain.divide(cos_zeni))

    # Net longwave radiation [MJ m-2 d-1] (FAO56 Eqn 39)
    rnl = tmax.expression(
        "4.901E-9 * ((Tmax ** 4 + Tmin ** 4) / 2) * (0.34 - 0.14 * sqrt(ea)) * " "(1.35 * (Rs / Rso) - 0.35)",
        {"Tmax": tmax.add(273.15), "Tmin": tmin.add(273.15), "ea": ea, "Rs": rs, "Rso": rso24h_dem},
    )

    # Net radiation [MJ m-2 d-1] (FAO56 Eqn 40)
    rn = tmax.expression("Rns - Rnl", {"Rns": rns, "Rnl": rnl})

    # Convert to W m-2
    rn = rn.multiply(11.6)

    return rn.rename("rad_24h")


def fexp_hot_pixel(
    time_start,
    albedo,
    ndvi,
    ndwi,
    lst,
    lst_dem,
    rn,
    g,
    tair,
    ux,
    ndvi_hot,
    lst_hot,
    geometry_image,
    coords,
    proj,
    dem,
    tfac,
    calibration_points=10,
):
    """
    Simplified CIMEC method to select the hot pixel


    Parameters
    ----------
    time_start : ee.Date
        Date information of the image.
    ndvi : ee.Image
        Normalized difference vegetation index.
    ndwi : ee.Image
        Normalized difference water index.
    lst : ee.Image
        Land surface temperature [K].
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
    proj : ee.Dictionary
        Landsat image projection.
    calibration_points : int
        Number of calibration points (the default is 10).

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
    ..
    """
    # Filtering only positive ndvi values
    pos_ndvi = ndvi.updateMask(ndvi.gt(0)).rename("post_ndvi")

    # Creating a negative ndvi raster for filtering
    ndvi_neg = pos_ndvi.multiply(-1).rename("ndvi_neg")
    
    # Creating a negative lst raster for filtering
    lst_neg = lst_dem.multiply(-1).rename("lst_neg")
    
    # Lst for non water pixels 
    lst_nw = lst_dem.updateMask(ndwi.lte(0)).rename("lst_nw")

    # Homogenetou mask for ndvi
    stdev_ndvi = homogeneous_mask(ndvi, proj)

    # Slope filter to avoid mountainous areas

    # Creating a raster with all the parameters
    images = pos_ndvi.addBands([ndvi, ndvi_neg, rn, g, pos_ndvi,
                                    lst_neg, lst_nw, lst, tair, ux, coords])
    
    # Estimating ndvi percentile [hottest]
    perc_low_ndvi = (
        images.select("post_ndvi")
        .updateMask(stdev_ndvi)
        .reduceRegion(reducer=ee.Reducer.percentile([ndvi_hot]), geometry=geometry_image, scale=30, maxPixels=1e9)
        .combine(ee.Dictionary({"post_ndvi": 100}), overwrite=False)
    )

    # Get top ndvi value
    perc_low_ndvi_value = ee.Number(perc_low_ndvi.get("post_ndvi"))

    # Filtering ndvi raster with the percecilte
    low_ndvi = images.updateMask(images.select("post_ndvi").lte(perc_low_ndvi_value))

    # Estimating lst percentile [hottest]
    perc_top_lst = (
        low_ndvi.select("lst_neg")
        .updateMask(stdev_ndvi)
        .reduceRegion(reducer=ee.Reducer.percentile([lst_hot]), geometry=geometry_image, scale=30, maxPixels=1e9)
        .combine(ee.Dictionary({"lst_neg": 350}), overwrite=False)
    )

    # Get low lst value
    perc_top_lst_value = ee.Number(perc_top_lst.get("lst_neg"))

    # Filtering lst raster with the percecilte  
    top_lst = low_ndvi.updateMask(stdev_ndvi)\
                        .updateMask(low_ndvi.select("lst_neg").lte(perc_top_lst_value))

    lst_hot_int = top_lst.select("lst_nw").min(1).max(1).int().rename("int")
    lst_hotpix = top_lst.addBands(lst_hot_int)

    lst_hotpix = lst_hotpix.select(
        ["ndvi", "rn_inst", "g_inst", "lst", "lst_nw", "tair", "ux", "longitude", "latitude", "int"]
    )

    sum_final_hot_pix = lst_hotpix.select("int").reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry_image, scale=30, maxPixels=1e9
    )
    sum_final_hot_pix_value = ee.Number(sum_final_hot_pix.get("int"))

    # CGM - Not used any more
    # def function_def_pixel(f):
    #     return f.setGeometry(ee.Geometry.Point([f.get('longitude'), f.get('latitude')]))
    
    # Creating a table with the cold pixels identified in the processed
    hot_pixels_table = lst_hotpix.addBands(tfac).stratifiedSample(
                numPoints=calibration_points,
                classBand="int",
                region=geometry_image,
                scale=30,
                dropNulls=True,
                geometries=True,
            )
    
    # Checkin if there are at least 1 pixel found as cold pixel
    minimum_cold_pixels = 1 #3000

    hot_pixels_table = ee.FeatureCollection(
        ee.Algorithms.If(sum_final_hot_pix_value.gt(minimum_cold_pixels),
            hot_pixels_table, 
            ee.FeatureCollection([ee.Feature(ee.Geometry.Point([0, 0]),
                                             {'ndvi': 0,'rn_inst': 0, 'g_inst': 0,
                                                    'lst_nw': 0, 'longitude': 0, 'latitude': 0, 'tfac':0})]))
    )
    
    return hot_pixels_table

def sensible_heat_flux(
    savi,
    ux,
    fc_cold_pixels,
    fc_hot_pixels,
    lst_dem,
    lst,
    dem,
    geometry_image,
    max_iterations=15,
):
    """
    Instantaneous Sensible Heat Flux [W m-2]

    Parameters
    ----------
    savi : ee.Image
        Soil-adjusted vegetation index.
    ux : ee.Image
        Wind speed [m s-1].
    fc_cold_pixels : ee.FeatureCollection
        Cold pixels.
    fc_hot_pixels : ee.FeatureCollection
        Hot pixels.
    lst_dem : ee.Image
        Land surface temperature (aspect/slope correction) [K].
    lst : ee.Image
        Land surface temperature [K].
    dem : ee.Image
        Digital elevation product [m].
    geometry_image : ee.Geometry
        Image geometry.
    max_iterations : int
        Maximum number of iterations (the default is 15).

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
    # Number of iterations to correct for instability
    iterations = ee.List.repeat(1, max_iterations)

    # Vegetation height [m]
    veg_height = ee.Number(0.5)

    # Wind speed height [m]
    zx = ee.Number(2)

    # Blending height [m]
    blending_height = ee.Number(200)

    # Air specific heat [J kg-1 K-1]
    cp = ee.Number(1004)

    # Von Karman’s constant
    k_constant = ee.Number(0.41)

    # Heights [m] above the zero plane displacement.
    z1 = ee.Number(0.01)
    z2 = ee.Number(2)

    # Filtering low lalues of wind speed
    wind_speed_std = (
        ux.rename("ux")
        .reduceRegion(reducer=ee.Reducer.stdDev(), geometry=geometry_image, scale=10000, maxPixels=1e9)
        .combine(ee.Dictionary({"ux": 0}), overwrite=False)
    )

    wind_speed_std_value = ee.Number(wind_speed_std.get("ux"))

    # LL : Values less than 1.5 m s-1 tend to generate instability in
    # the iterative process to estimate aerodynamic resistance.
    # Standard Deviation is added in this situations.
    ux = ux.where(ux.lt(1.5), ux.add(wind_speed_std_value))

    # Slope/ Aspect
    #slope_aspect = ee.Terrain.products(dem)

    # Momentum roughness length at the weather station. (Allen2002 Eqn 28)
    zom_first_approach = veg_height.multiply(0.123)

    # Friction velocity at the weather station. (Allen2002 Eqn 37)
    # TODO: LL - We need to change this approach
    ustar_station = lst.expression("(k_constant * ux) / log(zx / zom)", {
                "k_constant": k_constant, "zx": zx, "zom": zom_first_approach, "ux": ux})

    # Wind speed at blending height at the weather station.  (Allen2002 Eqn 29)
    u200 = lst.expression(
        "ustar_station * log(blending_height / zom) / k_constant",{
            "ustar_station": ustar_station, "blending_height": blending_height,
                "zom": zom_first_approach, "k_constant": k_constant})

    # Momentum roughness length for each pixel.
    #zom = lst.expression('exp((5.62 * savi) - 5.809)', {'savi': savi})
    zom = veg_height.multiply(0.123)

    # Momentum roughness slope/aspect Correction.  (Allen2002  A12 Eqn9)
    #zom = zom.expression(
    #     'zom * (1 + (slope - 5) / 20)',
    #     {'zom': zom, 'slope': slope_aspect.select('slope')}
    #)

    # Finding the a and b coefficients of  dt = a+bTs equation
    def map_cold(cold):
        
        # Cold pixel variables
        cold_pixel = ee.Feature(cold)

        # Lst cold pixel value
        ts_cold_pixel = ee.Number(cold_pixel.get("lst_nw")).float()

        # Mapping hot pixels
        def map_hot(hot):

            # Hot pixel variables
            hot_pixel = ee.Feature(hot)

            # Lst hot pixel value
            ts_hot_pixel = ee.Number(hot_pixel.get("lst_nw")).subtract(ee.Number(hot_pixel.get("tfac")))
            #n_Ts_true_hot = ee.Number(hot_pixel.get("lst"))

            # G inst hot pixel value
            g_inst_hot = ee.Number(hot_pixel.get("g_inst"))

            # Rn inst hot pixel value
            rn_inst_hot = ee.Number(hot_pixel.get("rn_inst"))

            # Wind speed hot pixel value
            ux_hot = ee.Number(hot_pixel.get("ux"))

            # Tair hot pixel value
            tair_hot = ee.Number(hot_pixel.get("tair")).add(273.15)
            # n_long_hot = ee.Number(f_hot.get("longitude"))
            # n_lat_hot = ee.Number(f_hot.get("latitude"))
            # p_hot_pix = ee.Geometry.Point([n_long_hot, n_lat_hot])

            # Density of the air [km m-2]
            #ro_hot = ts_hot_pixel.multiply(-0.0046).add(2.5538)

            # Sensible heat flux at the hot pixel
            h_inst_hot = ee.Number(rn_inst_hot).subtract(ee.Number(g_inst_hot))

            # Roughness at hot pixel []
            # TODO:LL - Why not the zoom at the hot pixel?
            zom_hot = zom_first_approach

            # Friction velocity at hot pixel [ ]
            ustar_hot = ux_hot.expression(
                "(k_constant * ux)/ log(zx /zom)", {
                    "k_constant": k_constant,
                    "zx": zx,
                    "zom": zom_hot,
                    "ux": ux_hot
                }
            )

            # Wind speed at blending height at 200 meters
            u200_hot = zom_hot.expression(
                "ustar_hot *  (log(blending_height/zom)/k_constant)",{
                    "ustar_hot": ustar_hot,
                    "blending_height": blending_height,
                    "zom": zom_hot,
                    "k_constant": k_constant
                }
            )

            # Filtering minimum value 
            u200_hot = u200_hot.max(ee.Number(4))

            # Aerodynamic resistance at hot pixel [s m-1]
            rah_hot = ustar_hot.expression(
                "(log(z2/z1))/(ustar_hot*0.41)",{
                    "z2": z2, "z1": z1, "ustar_hot": ustar_hot
                }
            )

            # Density of the air [km m-2]
            ro_hot = (ee.Number(-0.0046).multiply(tair_hot)).add(ee.Number(2.5538))

            # First feature of iterative process
            hot_pixel = ee.Feature(hot_pixel.geometry(),{
                "ustar": ustar_hot,"rah": rah_hot})

            # Iterative_process
            def iterative_hot(empty, old):
                
                feat = ee.Feature(old)

                # Aerodynamic resistance at the hot pixel [s m-1]
                rah_hot = ee.Number(feat.get("rah")).max(0)

                # Friction velocity at hot pixel []
                ustart_hot = ee.Number(feat.get("ustar")).max(0)

                # Monin obukov length at the hot pixel [m]
                l_hot = ee.Number(ustart_hot.expression(
                        "-(ro_hot*cp*(ustar_hot**3)*ts_hot_pixel)/(0.41*9.81*h_inst_hot)",
                        {
                            "ro_hot": ro_hot,
                            "cp": cp,
                            "ustar_hot": ustart_hot,
                            "ts_hot_pixel": ts_hot_pixel,
                            "h_inst_hot": h_inst_hot,
                        },
                    
                ))

                # Filtering minimum value 
                l_hot = l_hot.max(ee.Number(-1000))

                # Momentum and heat transport 
                # Stable conditions (l > 0)
                psim_200 = ee.Number(l_hot.expression("-5*(height/l_hot)", {"height": blending_height, "l_hot": l_hot}))
                psih_2 = ee.Number(l_hot.expression("-5*(height/l_hot)", {"height": z2, "l_hot": l_hot}))
                psih_01 = ee.Number(l_hot.expression("-5*(height/l_hot)", {"height": z1, "l_hot": l_hot}))

                # MO variables
                x200 = ee.Number(l_hot.expression("(1-(16*(height/l_hot)))**0.25", {"height": blending_height, "l_hot": l_hot}))
                
                x2 = ee.Number(l_hot.expression("(1-(16*(height/l_hot)))**0.25", {"height": z2, "l_hot": l_hot}))
                x01 = ee.Number(l_hot.expression("(1-(16*(height/l_hot)))**0.25", {"height": z1, "l_hot": l_hot}))

                psimu_200 = ee.Number(x200.expression(
                        "2*log((1+x200)/2)+log((1+x200**2)/2)-2*atan(x200)+0.5*pi",
                        {"x200": x200, "pi": ee.Number(math.pi)},
                    ))
                
                psihu_2 = ee.Number(x2.expression("2*log((1+x2**2)/2)", {"x2": x2}))

                psihu_01 = ee.Number(x01.expression("2*log((1+x01**2)/2)", {"x01": x01}))
                
                # Correcting atmospheric conditions
                #psim_200 = psim_200.where(l_hot.lt(0), psimu_200)
                #psim_200 = psim_200.where(l_hot.eq(0), 0)
               
                #psih_2 = psih_2.where(l_hot.lt(0), psihu_2)
                #psih_2 = psih_2.where(l_hot.eq(0), 0)

                #psih_01 = psih_01.where(l_hot.lt(0), psihu_01)
                #psih_01 = psih_01.where(l_hot.eq(0), 0)


                psim_200 = ee.Number(ee.Algorithms.If(l_hot.lt(0),psimu_200,
                        ee.Algorithms.If(l_hot.eq(0),ee.Number(0),psim_200))
                )
                psih_2 = ee.Number(ee.Algorithms.If(l_hot.lt(0),psihu_2,
                        ee.Algorithms.If(l_hot.eq(0),ee.Number(0), psih_2))
                )

                psih_01 = ee.Number(ee.Algorithms.If(l_hot.lt(0),psihu_01,
                        ee.Algorithms.If(l_hot.eq(0),ee.Number(0),psih_01  ))
                )

                # Friction velocity at the hot pixel
                ustar_hot = ee.Number(
                    u200_hot.expression(
                        "(u200*0.41)/(log(height/zom)-psim_200)",
                        {"u200": u200_hot, "height": blending_height, "zom": zom_hot, "psim_200": psim_200},
                    )
                )   

                # Filtering to avoid computing instability
                ustar_hot = ustar_hot.max(ee.Number(0.02))

                # Aerodynamic resistance to heat transport [s/m]
                rah_hot = ee.Number(ustar_hot.expression(
                        "(log(z2/z1)-psi_h2+psi_h01)/(ustar*0.41)",
                        {"z2": z2, "z1": z1, "ustar": ustar_hot, "psi_h2": psih_2, "psi_h01": psih_01},
                    ))

                return ee.Feature(hot_pixel.geometry(),{"ustar": ustar_hot,"rah": rah_hot})

            # Apply iterative function
            hot_pixel = ee.Feature(iterations.iterate(iterative_hot, hot_pixel))

            # Aerodynamic resistance to heat transport [s/m]
            rah_hot = ee.Number(hot_pixel.get("rah"))

            # Difference temperature at the hot pixel
            dT_hot = (h_inst_hot.multiply(rah_hot)).divide(ro_hot.multiply(cp))

            # Difference temperature at the cold pixel
            dT_cold = ee.Number(0)

            # Coef a and b
            coef_a = (dT_cold.subtract(dT_hot)).divide(ts_cold_pixel.subtract(ts_hot_pixel))
            coef_b = dT_hot.subtract(coef_a.multiply(ts_hot_pixel))

            #coef_b = coef_b.rename('b').reduceRegion(
            #    reducer = ee.Reducer.first(),
            #    scale=30,
            #    geometry=hot_pixel.geometry()
            #).getNumber('b')

            #coef_a = coef_a.rename('a').reduceRegion(
            #     reducer = ee.Reducer.first(),
            #   scale=30,
             #   geometry=hot_pixel.geometry()
            #).getNumber('a')

            return ee.Feature(None, {"a": coef_a, "b": coef_b})

        return fc_hot_pixels.toList(100).map(map_hot)


    # list length
    list_length = 100

    # Dictionary with the results
    dict_linear_coeffs = ee.FeatureCollection(fc_cold_pixels.toList(list_length).map(map_cold).flatten())

    # Coefs lists
    coeff_a_list = dict_linear_coeffs.aggregate_array("a")
    coeff_b_list = dict_linear_coeffs.aggregate_array("b")

    coeff_a_list = dict_linear_coeffs.aggregate_array("a")
    coeff_b_list = dict_linear_coeffs.aggregate_array("b")

    #coef_a = ee.Image(coeff_a_list.reduce(ee.Reducer.median()))
    #coef_a = ee.List([coef_a,0]).reduce(ee.Reducer.firstNonNull())
    #coef_a = ee.Image.constant(coef_a).float()
    coef_a = ee.Number(coeff_a_list.reduce(ee.Reducer.median()))

    #coef_b = ee.Image(coeff_b_list.reduce(ee.Reducer.median()))
    #coef_b = ee.List([coef_b,0]).reduce(ee.Reducer.firstNonNull())
   #coef_b = ee.Image.constant(coef_b).float()
    coef_b = ee.Number(coeff_b_list.reduce(ee.Reducer.median()))

    # Surface Temperature difference [K]
    dt = lst_dem.expression(
        "(coef_a * lst_dem) + coef_b", {"coef_a": coef_a, "coef_b": coef_b, "lst_dem": lst_dem}
    ).rename("dt")

    virtual_ta = lst_dem.expression("lst - dt", {"lst": lst, "dt": dt})

    # Density of the air [km m-2]
    ro = virtual_ta.expression("(-0.0046 * virtual_ta) + 2.5538", {"virtual_ta": virtual_ta}).rename("ro")

    # Iterative function to define rah and ustar
    def iterative_img(empty, img):

        img = ee.Image(img)

        # Sensible heat flux [W m-2]
        h_inst = img.expression(
            "(ro*cp*dt)/rah", {"ro": ro, "cp": cp, "dt": dt, "rah": img.select("rah")}
        ).rename("h_inst")
        
        # Monin Obukov length [m]
        mo_l = img.expression(
            "-(ro*cp*(ustar**3)*lst)/(0.41*9.81*h_inst)",
            {"ro": ro, "cp": cp, "ustar": img.select("ustar"), "lst": lst, "h_inst": h_inst},
        ).rename("ustar")

        # Filtering L to avoi numerical instability
        mo_l = mo_l.max(-1000)

        # Momentum and heat transport 
        # Stable conditions (mo_l > 0)
        psim_200 = mo_l.expression("-5*(height/mo_l)", {"height": ee.Number(200), "mo_l": mo_l})
        psih_2 = mo_l.expression("-5*(height/mo_l)", {"height": ee.Number(2), "mo_l": mo_l})
        psih_01 = mo_l.expression("-5*(height/mo_l)", {"height": ee.Number(0.1), "mo_l": mo_l})

        # Variables
        x200 = mo_l.expression("(1-(16*(height/mo_l)))**0.25", {"height": ee.Number(200), "mo_l": mo_l})
        x2 = mo_l.expression("(1-(16*(height/mo_l)))**0.25", {"height": ee.Number(2), "mo_l": mo_l})
        x01 = mo_l.expression("(1-(16*(height/mo_l)))**0.25", {"height": ee.Number(0.1), "mo_l": mo_l})

        # Momentumand heat transport
        # Unstable conditions (mo_l < 0)
        psimu_200 = x200.expression(
            "2*log((1+x200)/2)+log((1+x200**2)/2)-2*atan(x200)+0.5*pi",
            {"x200": x200, "pi": ee.Number(math.pi)},
        )

        psihu_2 = x2.expression("2*log((1+x2**2)/2)", {"x2": x2})

        psihu_01 = x01.expression("2*log((1+x01**2)/2)", {"x01": x01})

        # Correcting atmospheric conditions 
        psim_200 = psim_200.where(mo_l.lt(0), psimu_200)
        psim_200 = psim_200.where(mo_l.eq(0), 0)

        psih_2 = psih_2.where(mo_l.lt(0), psihu_2)
        psih_2 = psih_2.where(mo_l.eq(0), 0)
        
        psih_01 = psih_01.where(mo_l.lt(0), psihu_01)
        psih_01 = psih_01.where(mo_l.eq(0), 0)

        # Friction velocity [s/m]
        ustar = img.expression(
            "(u200*0.41)/(log(height/zom)-psim_200)",
            {"u200": u200, "height": blending_height, "zom": zom, "psim_200": psim_200},
        )

        # Filtering to avoid computing instability
        ustar = ustar.max(0.02).rename('ustar')  #where(u_star.lt(0.02), 0.02).rename("u_fr")

        # Aerodynamic resistance to heat transport [s/m]
        rah = img.expression(
            "(log(z2/z1)-psi_h2+psi_h01)/(ustar*0.41)",
            {"z2": z2, "z1": z1, "ustar": ustar, "psi_h2": psih_2, "psi_h01": psih_01},
        ).rename("rah")

        return ee.Image.cat([ustar, rah])

    # First approximation of Friction velocity [s/m]
    ustar = lst.expression(
        "(k_constant *u200) /(log(height/zom))", {
            "u200": u200, "height": blending_height, "zom": zom_first_approach, "k_constant": k_constant}
    ).rename("ustar")

    # First approcimation of aerodynamic resistance [s/m]
    rah = lst.expression("(log(z2/z1))/(ustar*0.41)", {"z2": z2, "z1": z1, "ustar": ustar}).rename("rah")

    # Preparing for the iteration stability process
    rah_ustar = ee.Image.cat([ustar, rah])

    # Get corrected aerodynamic resitance and velocity friction
    rah_ustart_corr = ee.Image(iterations.iterate(iterative_img, rah_ustar))

    # Final Sensible Heat estimation [W/m2]
    h_inst = lst.expression(
        "(ro*cp*dt)/rah", {"ro": ro, "cp": cp, "dt": dt, "rah": rah_ustart_corr.select("rah")}
    ).rename("h_inst")

    return h_inst.rename("h_inst")


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
    le_inst = h_inst.expression("(rn_inst - g_inst - h_inst)", {"rn_inst": rn_inst, "g_inst": g_inst, "h_inst": h_inst})

    # Latent heat of vaporization or the heat
    # absorbed when a kilogram of water evaporates [J/kg].
    lambda_et = h_inst.expression("(2.501 - 0.002361 * (lst - 273.15))", {"lst": lst_dem})

    # Evaporative fraction
    ef = h_inst.expression("le_inst / (rn_inst - g_inst)", {"le_inst": le_inst, "rn_inst": rn_inst, "g_inst": g_inst})

    # clamping between 0 and 1
    ef = ef.clamp(0, 1)

    # Caculating ET using Rn24h 
    daily_et = ef.expression(
        "(0.0864 * ef * rad_24h) / lambda_et", {"ef": ef, "lambda_et": lambda_et, "rad_24h": rad_24h}
    )

    return daily_et.rename("et")


def et_fraction(time_start, et, et_reference_source, et_reference_band, et_reference_factor):
    """ET Fraction

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

    References
    ----------
    """
    date = ee.Date(time_start)
    start_date = ee.Date(utils.date_to_time_0utc(date))

    eto = (
        ee.ImageCollection(et_reference_source)
        .select(et_reference_band)
        .filterDate(start_date, start_date.advance(1, "day"))
    )
    et_reference_img = ee.Image(eto.first())
    et_reference_img = et_reference_img.multiply(et_reference_factor)

    et_fraction = et.divide(et_reference_img).rename("et_fraction")

    return et_fraction

