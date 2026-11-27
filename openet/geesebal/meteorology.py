import ee


def nldas_gridmet(time_start, meteorology_source_inst, meteorology_source_daily):
    """
    Load NLDAS-2 and GRIDMET meteorology for the target image time

    Parameters
    ----------
    time_start : ee.Number, int, str
        Image property: time start of the image.
    meteorology_source_inst : ee.ImageCollection, str
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
    """
    # Get date information
    time_start = ee.Number(time_start)

    # Filtering Daily data
    meteorology_daily = (
        ee.ImageCollection(meteorology_source_daily)
        .filterDate(ee.Date(time_start).advance(-1, 'day'), ee.Date(time_start))
        .first()
    )

    # Instantaneous data
    meteorology_inst_collection = ee.ImageCollection(meteorology_source_inst)

    # Linear interpolation
    previous_time = time_start.subtract(2 * 60 * 60 * 1000)
    next_time = time_start.add(2 * 60 * 60 * 1000)

    previous_image = (
        meteorology_inst_collection.filterDate(previous_time, time_start)
        .limit(1, 'system:time_start', False).first()
    )

    next_image = (
        meteorology_inst_collection.filterDate(time_start, next_time)
        .limit(1, 'system:time_start', True).first()
    )

    image_prev_time = ee.Number(previous_image.get('system:time_start'))
    image_next_time = ee.Number(next_image.get('system:time_start'))
    delta_time = time_start.subtract(image_prev_time).divide(image_next_time.subtract(image_prev_time))

    # Incoming shortwave down [W m-2]
    rs_24h = meteorology_daily.select(['srad'], ['rs_24h'])

    # Minimum air temperature [K]
    tmin = meteorology_daily.select(['tmmn'], ['tmin'])

    # Maximum air temperature [K]
    tmax = meteorology_daily.select(['tmmx'], ['tmax'])

    # Instantaneous short wave radiation [W m-2]
    rs_inst = (
        next_image.select('shortwave_radiation')
        .subtract(previous_image.select('shortwave_radiation'))
        .multiply(delta_time)
        .add(previous_image.select('shortwave_radiation'))
        .rename('rs_inst')
    )

    # Specific humidity [Kg Kg-1]
    q_med = (
        next_image.select('specific_humidity')
        .subtract(previous_image.select('specific_humidity'))
        .multiply(delta_time)
        .add(previous_image.select('specific_humidity'))
    )

    # Air temperature [C]
    tair = (
        next_image.select('temperature')
        .subtract(previous_image.select('temperature'))
        .multiply(delta_time)
        .add(previous_image.select('temperature'))
        .rename('tair')
    )

    # Wind speed u [m s-1]
    wind_u = (
        next_image.select('wind_u')
        .subtract(previous_image.select('wind_u'))
        .multiply(delta_time)
        .add(previous_image.select('wind_u'))
    )

    # Wind speed u [m s-1]
    wind_v = (
        next_image.select('wind_v')
        .subtract(previous_image.select('wind_v'))
        .multiply(delta_time)
        .add(previous_image.select('wind_v'))
    )

    wind_med = wind_u.expression(
        'sqrt(ux_u ** 2 + ux_v ** 2)', {'ux_u': wind_u, 'ux_v': wind_v}
    )

    # Wind speed [m s-1] (FAO56 Eqn 47)
    wind_med = wind_med.expression(
        'ux * (4.87) / log(67.8 * z - 5.42)', {'ux': wind_med, 'z': 10.0}
    ).rename('ux')

    # Pressure [kPa]
    p_med = (
        next_image.select('pressure')
        .subtract(previous_image.select('pressure'))
        .multiply(delta_time)
        .add(previous_image.select('pressure'))
        .divide(ee.Number(1000))
    )

    # Actual vapor pressure [kPa] (Shuttleworth Eqn 2.10)
    ea = p_med.expression('(1 / 0.622) * Q * P', {'Q': q_med, 'P': p_med})

    # Saturated vapor pressure [kPa] (FAO56 Eqn 11)
    esat = tair.expression(
        '0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))', {'T_air': tair}
    )

    # Relative humidity (%)  (FAO56 Eqn 10)
    rh = ea.divide(esat).multiply(100).rename('rh')

    # Surface temperature correction based on precipitation and reference ET

    # Accumulation time period
    accum_period = -60

    # Accum meteo data
    gridmet_accum = (
        ee.ImageCollection(meteorology_source_daily)
        .filterDate(ee.Date(time_start).advance(accum_period, 'days'), ee.Date(time_start))
    )

    # Reference ET
    etr_accum = gridmet_accum.select('etr').sum()

    # Precipitation
    precipt_accum = gridmet_accum.select('pr').sum()

    # Ratio between precipt/etr
    ratio = precipt_accum.divide(etr_accum)

    # Temperature adjustment offset (Allen2013 Eqn 8)
    tfac = etr_accum.expression('2.6 - 13 * ratio', {'ratio': ratio})
    tfac = ee.Image(tfac.where(ratio.gt(0.2), 0)).rename('tfac')

    # Wind velocity correction (limit to 1.5)
    # wind_clamp = wind_med.max(1.5).rename('ux_clamp')

    # Resample
    tmin = tmin.subtract(273.15).resample('bilinear')
    tmax = tmax.subtract(273.15).resample('bilinear')
    tair = tair.resample('bilinear')
    wind_med = wind_med.resample('bilinear')
    rh = rh.resample('bilinear')
    rs_inst = rs_inst.resample('bilinear')
    rs_24h = rs_24h.resample('bilinear')
    # wind_clamp = wind_clamp.resample('bilinear')

    return [tmin, tmax, tair, wind_med, rh, rs_inst, rs_24h, tfac]


def era5land(time_start, meteorology_source_inst, meteorology_source_daily):
    """
    Load ERA5-Land meteorology for the target image time

    Parameters
    ----------
    time_start : ee.Number, int, str
        Image property: time start of the image.
    meteorology_source_inst : ee.ImageCollection, str
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
    """

    # Get date information
    time_start = ee.Number(time_start)

    # Filtering Daily data
    meteorology_daily = (
        ee.ImageCollection(meteorology_source_daily)
        .filterDate(ee.Date(time_start), ee.Date(time_start).advance(1, 'day'))
        .first()
    )

    # Instantaneous data
    meteorology_inst_collection = ee.ImageCollection(meteorology_source_inst)

    # Linear interpolation
    previous_time = time_start.subtract(1 * 60 * 60 * 1000)
    next_time = time_start.add(1 * 60 * 60 * 1000)

    previous_image = (
        meteorology_inst_collection.filterDate(previous_time, time_start)
        .limit(1, 'system:time_start', False).first()
    )

    next_image = (
        meteorology_inst_collection.filterDate(time_start, next_time)
        .limit(1, 'system:time_start', True).first()
    )

    image_prev_time = ee.Number(previous_image.get('system:time_start'))
    image_next_time = ee.Number(next_image.get('system:time_start'))

    delta_time = time_start.subtract(image_prev_time).divide(image_next_time.subtract(image_prev_time))

    # Incoming shortwave down [W m-2]
    # The band is called "surface_solar_radiation_downwards_sum" in ECMWF/ERA5_LAND/DAILY_AGGR
    rs_24h = (
        meteorology_daily.select(['surface_solar_radiation_downwards'], ['rs_24h'])
        # meteorology_daily.select(['surface_solar_radiation_downwards_sum'], ['rs_24h'])
        .divide(1 * 60 * 60 * 24)
    )

    # Minimum air temperature [K]
    tmin = meteorology_daily.select(['temperature_2m_min'], ['tmin'])

    # Maximum air temperature [K]
    tmax = meteorology_daily.select(['temperature_2m_max'], ['tmax'])

    # Instantaneous incoming shortwave radiation [W m-2]
    rs_inst = (
        ee.ImageCollection(meteorology_source_inst)
        .filterDate(ee.Date(time_start), ee.Date(time_start).advance(1, 'hour'))
        .select('surface_solar_radiation_downwards_hourly')
        .mean()
        .divide(1 * 60 * 60)
        .rename('rs_inst')
    )

    # Air temperature [C]
    # TODO: LL- Change all temperatures to K ?
    tair = (
        next_image.select('temperature_2m')
        .subtract(previous_image.select('temperature_2m'))
        .multiply(delta_time)
        .add(previous_image.select('temperature_2m'))
        .subtract(273.15)
        .rename('tair')
    )

    # Wind speed [m/s]
    wind_u = (
        next_image.select('u_component_of_wind_10m')
        .subtract(previous_image.select('u_component_of_wind_10m'))
        .multiply(delta_time)
        .add(previous_image.select('u_component_of_wind_10m'))
    )

    wind_v = (
        next_image.select('v_component_of_wind_10m')
        .subtract(previous_image.select('v_component_of_wind_10m'))
        .multiply(delta_time)
        .add(previous_image.select('v_component_of_wind_10m'))
    )

    wind_med = wind_u.expression(
        'sqrt(ux_u ** 2 + ux_v ** 2)', {'ux_u': wind_u, 'ux_v': wind_v}
    )

    wind_med = wind_med.expression(
        'ux * (4.87) / log(67.8 * z - 5.42)', {'ux': wind_med, 'z': 10.0}
    ).rename('ux')

    # Dew point temperature [°K]
    tdp = (
        next_image.select('dewpoint_temperature_2m')
        .subtract(previous_image.select('dewpoint_temperature_2m'))
        .multiply(delta_time)
        .add(previous_image.select('dewpoint_temperature_2m'))
        .rename('tdp')
    )

    # Actual vapour pressure [kPa]
    ea = tdp.expression(
        '0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))', {'T_air': tdp.subtract(273.15)}
    )

    # SATURATED VAPOR PRESSURE [kPa]
    esat = tair.expression('0.6108 * (exp((17.27 * T_air) / (T_air + 237.3)))', {'T_air': tair})

    # RELATIVE HUMIDITY (%)
    rh = ea.divide(esat).multiply(100).rename('rh')

    # Surface temperature correction based on precipitation and reference ET
    # Accumulation time period
    accum_period = -60

    # Accum meteo data
    # Precipitation
    precipt_accum = (
        ee.ImageCollection(meteorology_source_daily)
        .filterDate(ee.Date(time_start).advance(accum_period, 'days'), ee.Date(time_start))
        .select('total_precipitation_sum').sum().multiply(1000).rename('total_precipitation')
    )

    # Reference ET
    # TODO: Switch to global reference ET or
    etr_accum = (
        ee.ImageCollection('projects/ee-openetbrazil/assets/meteorological/daily/era5-land/sa')
        .filterDate(ee.Date(time_start).advance(accum_period, 'days'), ee.Date(time_start))
        .select('etr_asce').sum()
    )

    # Ratio between precipt/etr
    ratio = precipt_accum.divide(etr_accum).rename('ratio')

    # Temperature adjustment offset (Allen2013 Eqn 8)
    tfac = etr_accum.expression('2.6 - 13 * ratio', {'ratio': ratio})
    tfac = ee.Image(tfac.where(ratio.gt(0.2), 0)).rename('tfac')

    # Wind velocity correction (limit to 1.5)
    # wind_clamp = wind_med.max(1.5).rename('ux_clamp')

    # Resample
    tmin = tmin.subtract(273.15).resample('bilinear')
    tmax = tmax.subtract(273.15).resample('bilinear')

    tair = tair.resample('bilinear')
    wind_med = wind_med.resample('bilinear')
    rh = rh.resample('bilinear')
    rs_inst = rs_inst.resample('bilinear')
    rs_24h = rs_24h.resample('bilinear')
    # wind_clamp = wind_clamp.resample('bilinear')

    return [tmin, tmax, tair, wind_med, rh, rs_inst, rs_24h, tfac]
