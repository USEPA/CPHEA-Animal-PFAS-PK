Explanation of columns for extracted data
=========================================

species: Dosed species for PFAS exposure
	options: rat, mouse, primate
	
animal_id: Unique animial identifier if individual animal is measured. If summary data, use -1

N_animals: Number of animals used for reported concentration. If indiviudal data, use 1

BW: Reported body weight for individual animal. If summary data, use record -1 and the study-level body weight when registering the study will be used.

BW_units: Body weight units
	options: kg, g

strain: Strain of animal
	options: sprague-dawley, CD-1, cynomolgus

dose: Applied dose given to animal

dose_units: Units on applied dose

time: Reported time from concentration-response data

time_units: Time units on reported time

conc_mean: Reported mean in time-course concentration plot (usually digitized)
	optional: -1 if mean is missing or below LoD

conc_sd: Reported standard deviation or standard error reported for the corresponding 'conc_mean'
	optional: -1 if mean is missing or below LoD
	
standard_error: Flag on if the digitized reported error is a standard error (1) or standard deviation (0).

conc_units: Concentration units

route: route of exposure
	options: gavage, iv

matrix: Tissue matrix that the concentrations were measured in

sex: Animal sex (if known). If unknown, use NR (Not reported)

dtxsid: DSSTOX ID for the PFAS used

source: Source of the concentration-resonse data in the hero ID, e.g. Table 1, Figure 3, etc.
