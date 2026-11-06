OpenET - geeSEBAL
=================

<img src="https://github.com/et-brasil/geeSEBAL/blob/master/Images/geeSEBAL_logo_update_cut.png?raw=true" width="140">

## Estimating Evapotranspiration using SEBAL model in Google Earth Engine platform.

* The Google Earth Engine Surface Energy Balance for Land (geeSEBAL) solves the energy balance equation (LE + H = Rn - G) to estimate Daily Evapotranspiration (ET) on a global scale by using Landsat images (L4, L5, L7, L8, and L9) and meteorological data (air temperature, relative humidity, global radiation and wind speed).

## Input Collections

* The following Earth Engine image collection are use in geeSEBAL:

| Image Collections IDs  |
| :--------------------: |
| LANDSAT/LC09/C02/T1_L2 |
| LANDSAT/LC08/C02/T1_L2 |
| LANDSAT/LE07/C02/T1_L2 |
| LANDSAT/LT05/C02/T1_L2 |
| LANDSAT/LT04/C02/T1_L2 | 

## Model Description

* Surface Energy Balance Algorithm for Land (SEBAL) was developed and validated by Bastiaanssen (Bastiaanssen et al., 1998a, 1998b) to estimate evapotranspiration (ET) from energy balance equation (Rn – G = LE + H), where LE, Rn, G and H are Latent Heat Flux, Net Radiation, Soil Heat Flux and Sensible Heat Flux, respectively. SEBAL estimates LE as a residual of others energy fluxes (LE = Rn - LE - G).
* SEBAL algorithm has an internal calibration, assuming a linear relationship between dT and LST across domain area, where dT is designed as a vertical air temperature (Ta) floating over the land surface, considering two extreme conditions. At the hot and dry extreme condition, LE is zero and H is equal to the available energy, whereas at the cold and wet extreme condition, H is zero and LE is equal to the available energy.
* Workflow of geeSEBAL, demonstrating remote sensing and global meteorological inputs, as well as data processing to estimate daily evapotranspiration. The version 1.0 includes ECMWF ERA5-Land hourly and daily reanalysis datasets for global applications. 

![fluxogram_openet_geesebal](https://user-images.githubusercontent.com/45111381/127649854-db066c12-8eb4-497c-8a4b-bed1791117d2.jpg)

## Model Design

### Image()

* Compute Daily ET or ET fraction for a single input image.
* Allow to obtain ET image collections by mapping over Landsat collections.

#### Landsat Collection 2 Input Image

* Select Image.from_landsat_c2_sr() method to instantiate the class for a Landsat Collection 2 SR image. Image must have the following bands and properties:

| SPACECRAFT_ID | Band Names                                                        |
| ------------- | ----------------------------------------------------------------- |
| **LANDSAT_4** | SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B7, ST_B6, QA_PIXEL         |
| **LANDSAT_5** | SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B7, ST_B6, QA_PIXEL         | 
| **LANDSAT_7** | SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B7, ST_B6, QA_PIXEL         | 
| **LANDSAT_8** | SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, ST_B10, QA_PIXEL | 
| **LANDSAT_9** | SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, ST_B10, QA_PIXEL | 

| PROPERTIES                                                                                    |
| --------------------------------------------------------------------------------------------- |
| **system: index** - Landsat scene ID (ex: LC08_044033_20170801)                               |
| **system: time_start** - Time start of the image in epoch time                                |
| **SPACECRAFT_ID** - Landsat Satellite (LANDSAT_4, LANDSAT_5, LANDSAT_7, LANDSAT_8, LANDSAT_9) |
| **SUN_ELEVATION** - Solar elevation angle in degrees                                          |

## Model Output

The general outputs of the geeSEBAL are NDVI (Normalized Difference Vegetation Index), LST (Land Surface Temperature), et_fraction and ET. They can be selected as example below:

### Example

	import openet.geesebal as geesebal
	
	ls_img = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044033_20170801')
	model_obj = geesebal.from_landsat_c2_sr(ls_img)

	ndvi = model_obj.ndvi
	lst = model_obj.lst
	et_fraction = model.et_fraction
	et = model_obj.et

## Examples Notebooks

Examples of how to use geeSEBAL model are detailed in *examples* folder:

[geeSEBAL examples.](https://github.com/et-brasil/openet-geesebal/blob/main/examples "Examples")

## Installation

	pip install openet-geesebal

### Depedencies

 * `earthengine-api` <https://github.com/google/earthengine-api>`
 * `openet-core` <https://github.com/Open-ET/openet-core>`

## References

Main Reference

[[2021] Laipelt, L., Kayser, R. H. B., Fleischmann A., Ruhoff, A., Bastiaanssen, W., Erickson, T., Melton, F. Long-term monitoring of evapotranspiration using the SEBAL algorithm and Google Earth Engine cloud computing. ISPRS J. Photogramm. Remote Sens. 178, p. 81-96.](https://doi.org/10.1016/j.isprsjprs.2021.05.018)

Additional References

[[2026] Rossi, J. B., Fleischmann, A. S., Laipelt, L., Comini de Andrade, B., Fisher, J. B., Huntington, J. L., Pearson, C., Woolway, R. I., Vale, R., Santana, R, A. S., Tota, J., Senay, G. B., Gao, H., Ruhoff, A. Global performance of remote sensing‐based and reanalysis‐driven models to estimate open water evaporation. Water Resour. Res., 62, e2025WR042363.](https://doi.org/10.1029/2025WR042363)

[[2024] Laipelt, L., Rossi, J.B., de Andrade, B.C., Scherer-Warren, M., Ruhoff, A. Assessing Evapotranspiration Changes in Response to Cropland Expansion in Tropical Climates. Remote Sens. 16, 3404.](https://doi.org/10.3390/rs16183404)

[[2022] Kayser, R.H., Ruhoff, A., Laipelt, L., Kich, E. de M., Roberti, D.R., Souza, V. de A., Rubert, G.C.D., Collischonn, W., Neale, C.M.U., 2022. Assessing geeSEBAL automated calibration and meteorological reanalysis uncertainties to estimate evapotranspiration in subtropical humid climates. Agric For Meteorol. 314, 108775.](https://doi.org/10.1016/j.agrformet.2021.108775)

[[2022] Gonçalves, I.Z., Ruhoff, A., Laipelt, L., Bispo, R.C., Hernandez, F.B.T., Neale, C.M.U., Teixeira, A.H.C., Marin, F.R., 2022. Remote sensing-based evapotranspiration modeling using geeSEBAL for sugarcane irrigation management in Brazil. Agric. Water Manag. 274, 10796.](https://doi.org/10.1016/j.agwat.2022.107965)
