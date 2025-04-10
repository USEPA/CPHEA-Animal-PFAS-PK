# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:14:34 2021

@author: Todd Zurlinden
"""

import sqlite3 as sql
from sqlite3 import Error
import pandas as pd
import numpy as np
import scipy.interpolate as sci
import os

class UnitsException(Exception):
    def __init__(self, message):
        self.message = message
    def __repr__(self):
        return self.message

class DataException(Exception):
    def __init__(self, message):
        self.message = message
    def __repr__(self):
        return self.message

class PFAS:   
    def __init__(self, db_file: str, pfas_file = '../auxiliary/pfas_master.csv', rat_growth_path='../auxiliary/rat_growth_NTP.csv', 
        correct_BW = True, scale=False, skip_hero = [], hero_only=None, route_only=None, skip_matrix = ['SI', 'LI'], 
        fit_only_blood = True, remove_single_times=True, censor=False):
        """Class to set up SQLite database and then read/write the data into the database. Once intialized, this class can be used to 
        get chemical/species/sex specific data from the database and then prepare the data for analysis.

        Parameters
        ----------
        db_file : str
            location of the SQLite database
        pfas_file : str, optional
            Master list of PFAS and their corresponding identifiers, by default '../auxiliary/pfas_master.csv'
        rat_growth_path : str, optional
            Strain- and sex-specific rat growth for interpolation, by default '../auxiliary/rat_growth.csv'
        correct_BW : bool, optional
            Correct body weights to be in units of Kg, by default True
        scale : bool, optional
            Scale the concentration data for a dataset by Cmax, by default False
        skip_hero : list, optional
            List of HERO IDs to skip when doing analysis, by default []
        hero_only : _type_, optional
            If only one HERO ID is desired, enter it here, e.g. hero_only=12345, by default None
        route_only : _type_, optional
            Only fit data for a specific route of exposure, e.g. route_only='iv' or route_only='oral' (all lower case), by default None
        skip_matrix : list, optional
            If fitting tissues, skip the tissues in this list. Moot if fit_only_blood=True, by default ['SI', 'LI']
        fit_only_blood : bool, optional
            Only fit blood (or serum) concentrations, by default True
        remove_single_times : bool, optional
            Remove values for if there is only one measurement for a hero_id/species/dose/matrix/route combination, by default True
        censor : bool, optional
            Depricated. Functionality for one day using using censored data, but setting to True will break, by default False
        """        
        
        self.db_file = db_file
        self.pfas_file = pfas_file
        self.rat_growth = pd.read_csv(rat_growth_path, index_col=0, header=None).T
        self.correct_BW = correct_BW
        self.scale = scale
        self.skip_hero = skip_hero
        self.hero_only = hero_only
        self.route_only = route_only
        self.skip_matrix = skip_matrix
        self.fit_only_blood = fit_only_blood
        self.remove_single_times = remove_single_times
        self.censor = censor

        # PFAS MW (CompTox)
        self.MW = {'PFHxA': 314.054,
                   'PFNA': 464.078,
                   'PFDA': 514.086,
                   'PFHxS': 400.11,
                   'PFBS': 300.09,
                   'PFBA': 214.039,
                   'PFOS': 500.13,
                   'PFOA': 414.07,                   
                   }
        
        
        # Private functions
        if not os.path.exists(self.db_file):
            self._write_initial_tables()
            self._populate_pfas()
    
    def _write_initial_tables(self):
        
        """If the SQLite table does not exist, create it"""
        
        # "study" is grouped by (hero_id, species, sex)
        study_table_sql = """
        CREATE TABLE IF NOT EXISTS `studies` (
          `study_id` INTEGER PRIMARY KEY,
          `hero_id` INT NOT NULL,
          `species` VARCHAR(45) NULL,
          `sex` VARCHAR(45) NULL,
          `initial_age` INT NULL,
          `initial_age_units` VARCHAR(45) NULL,
          `initial_weight` FLOAT NULL,
          `initial_weight_units` VARCHAR(45) NULL,
          `n` INT NULL,
          `author` VARCHAR(45) NULL
          );"""
        
        lod_table_sql = """
        CREATE TABLE IF NOT EXISTS `lod` (
            `hero_id` INT NOT NULL,
            `pfas_abbrev` VARCHAR(45) NULL,
            `lod` FLOAT NULL,
            `lod_units` VARCHAR(45) NULL
        );"""
        
        data_table_sql = """
        CREATE TABLE IF NOT EXISTS `data` (
          `hero_id` INT NOT NULL,
          `study_id` INT NOT NULL,
          `animal_id` INT NULL,
          `N_animals` INT NULL,
          `BW` FLOAT NULL,
          `BW_units` VARCHAR(45) NULL,
          `BW_init` FLOAT NULL,
          `BW_init_units` VARCHAR(45) NULL,
          `species` VARCHAR(45) NULL,
          `strain` VARCHAR(45) NULL,
          `dose` FLOAT NULL,
          `dose_units` VARCHAR(45) NULL,
          `time` FLOAT NULL,
          `time_units` VARCHAR(45) NULL,
          `conc_mean` FLOAT NULL,
          `conc_sd` FLOAT NULL,
          `standard_error` INT NULL,
          `conc_units` VARCHAR(45) NULL,
          `route` VARCHAR(45) NULL,
          `matrix` VARCHAR(45) NULL,
          `sex` VARCHAR(45) NULL,
          `dtxsid` INT NULL,
          `source` VARCHAR(45) NULL
          );
        """
        
        pfas_table_sql = """
        CREATE TABLE IF NOT EXISTS `pfas` (
          `pfas_id` INTEGER PRIMARY KEY,
          `preferred_name` VARCHAR(45) NULL,
          `pfas_abbrev` VARCHAR(45) NULL,
          `casrn` VARCHAR(45) NULL,
          `dtxsid` VARCHAR(45) NULL,
          `iupac_name` VARCHAR(45) NULL,
          `mapped_dtxsid` VARCHAR(45) NULL
          
          );
        """
        
        half_lives_table_sql = """
        CREATE TABLE IF NOT EXISTS `reported_half_lives` (
        `hero_id` INT NOT NULL,
        `species` VARCHAR(45) NULL,
        `sex` VARCHAR(45) NULL,
        `dtxsid` INT NOT NULL,
        `route` VARCHAR(45) NULL,
        `dose` FLOAT NULL,
        `matrix` VARCHAR(45) NULL,
        `half_life` FLOAT NULL,
        `units` VARCHAR(45) NULL,
        `method` VARCHAR(45) NULL,
        `study_id` INT NOT NULL
        );
        """
        for table_sql in [study_table_sql, lod_table_sql, data_table_sql, pfas_table_sql, half_lives_table_sql]:
            self._create_table(table_sql)

    def _populate_pfas(self):
        pfas_master = pd.read_csv(self.pfas_file)
        conn = self._connect()
        #SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';
        pfas_master.to_sql('pfas', conn, if_exists='replace', index=False)
      
    def _connect(self):
        conn = None
        try:
            conn = sql.connect(self.db_file)
            return conn
        except Error as e:
            print(e)
        return conn
    
    def _close_connection(self):
        conn = self._connect()
        try:
            c = conn.cursor()
            c.close()
        except Error as e:
            print(e)
    
    def _scale_data(self, conc: np.array):
        return np.max(conc), conc/np.max(conc)
    
    def _create_table(self, create_table_sql: str):
        """
        Create table from sql statement
        
        Parameters
        ----------
        create_table_sql : str
            a CREATE TABLE statement

        Returns
        -------
        None.

        """
        
        conn = self._connect()
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)

    def send_query(self, sql: str):
        """
        Parameters
        ----------
        sql : str
            sql query to pcbs DB.

        Returns
        -------
        queried dataframe.

        """
        
        conn = self._connect()
        df = pd.read_sql_query(sql, conn)
        df.replace([None], np.nan, inplace=True)
        return df
    
    def get_data(self, pfas_abbrev: str, study=False):
        """
        create pandas dataframe with all data for given pcb number

        Parameters
        ----------
        pfas : int
            pcb congener number

        Returns
        -------
        pandas dataframe with pcb data

        """
        
        if study and self.censor:
            sql = """SELECT * from data JOIN studies using (hero_id, sex, species) JOIN pfas using (dtxsid) JOIN lod using (hero_id, pfas_abbrev) WHERE pfas_abbrev="%s" """%pfas_abbrev
        elif study and not self.censor:
            sql = """SELECT * from data JOIN studies using (hero_id, sex, species) JOIN pfas using (dtxsid) WHERE pfas_abbrev="%s" """%pfas_abbrev
        else:
            sql = """SELECT * from data JOIN pfas using (dtxsid) WHERE pfas_abbrev="%s" """%pfas_abbrev    
        return self.send_query(sql)
    
    def get_study_id(self, hero_id, species, sex):
        sql_query = f"SELECT study_id from studies WHERE hero_id={hero_id} AND species='{species}' AND sex='{sex}'"
        study_id = self.send_query(sql_query).loc[0, 'study_id']
        return study_id
    
    def get_dtxsid(self, pfas_abbrev):
        sql_query = f"SELECT dtxsid from pfas WHERE pfas_abbrev='{pfas_abbrev}'"
        dtxsid = self.send_query(sql_query).loc[0, 'dtxsid']
        return dtxsid
    
    def register_study(self, species: str, sex: str, n: int, age:float, 
                       age_units: str, weight: float, 
                       weight_units:str, author:str, hero_id:int):
        """
        Register a new study with PCB pk data

        Parameters
        ----------
        species: str
            Species used for study
        sex: str
            Animal sex for study
        n : int
            Number of animals used for each data point.
        age: int
            Age of rats at beginning of the study
        age_units: str
            Units of rat age at beginning of study
        weight: int
            Initial weight of rats at beginning of study
        weight_units: str
            Units of rat weight at beginning of study
        author: str
            Lead author of study
        hero_id : int
            HERO id for study.

        Returns
        -------
        None.

        """
        
        study_info = {'species': species,
                      'n': n,
                      'sex': sex,
                      'initial_age': age,
                      'initial_age_units': age_units,
                      'initial_weight': weight,
                      'initial_weight_units': weight_units,
                      'author': author,
                      'hero_id': hero_id}
        
        df = pd.DataFrame(pd.Series(study_info)).T 
        conn = self._connect()
        df.to_sql('studies', con=conn, if_exists='append', index=False)

    def _convert_abbrev(self, row):
        return self.get_dtxsid(row.pfas_abbrev)

    def register_reported_lod(self, lod_dict: dict, hero_id: int, lod_units: str):
        """
        Register the limit of detection for a PFAS in a given study (hero_id)
        
        Parameters
        ----------
        hero_id : int
            HERO ID for the study
        lod_units: str
            Concentration units for lod
        lod_dict : dict
            dictionary containing the PFAS lod's for a given study
            Example: half_life_dict = {PFAS_abbrev1: lod1,
                                       PFAS_apprev2: lod2}
        """


        write_df = pd.DataFrame.from_dict(lod_dict, orient='index', columns=['lod'])
        write_df.index.name = 'pfas_abbrev'
        write_df = write_df.reset_index()
        write_df['hero_id'] = hero_id
        write_df['lod_units'] = lod_units
        conn = self._connect()
        write_df.to_sql('lod', con=conn, if_exists='append', index=False)

    def register_reported_half_lives(self, half_life_dict: dict, species:str, sex: str, route: str,
                                     hero_id: int, dose: float, units='hours', method='2-compartment'):
        global write_df
        """
        Register half-lives reported in the study

        Parameters
        ----------
        hero_id : int
            HERO ID for the study
        species: str
            Species for half-life dictionary
        sex: str
            Species sex corresponding to half-life dict
        dose: float
            Dose corresponding to the reported half-lives
        route: str
            Route of exposure for half-life
        half_life_dict : dict
            dictionary containing the pcbs, matrices, and reported half-lives
            Example: half_life_dict = {PFAS_abbrev: {matrix: half_life}}

        Returns
        -------
        None.

        """
        
        
        df = pd.DataFrame(half_life_dict)
        df['matrix'] = df.index
        write_df = df.melt(value_vars=list(half_life_dict.keys()), id_vars=['matrix'], var_name='pfas_abbrev', value_name='half_life')
        write_df.dropna(inplace=True, axis=0)
        
        write_df['dtxsid'] = write_df.apply(self._convert_abbrev, axis=1)
        write_df.drop(columns=['pfas_abbrev'], inplace=True)
        write_df['hero_id'] = hero_id
        write_df['species'] = species
        write_df['sex'] = sex
        write_df['units'] = units
        write_df['method'] = method
        write_df['route'] = route
        study_id = self.get_study_id(hero_id, species, sex)
        write_df['study_id'] = study_id
        write_df['dose'] = dose
        conn = self._connect()
        write_df.to_sql('reported_half_lives', con=conn, if_exists='append', index=False)

    def register_data(self, data_path: str, hero_id: int):
        global df
        """
        Register the data for a particular hero id

        Parameters
        ----------
        data_path : str
            Path to preprocessed csv containing the time, conc_mean, conc_sd,
            units, matrix, pcb_id
        hero_id : int
            HERO id for study.

        Returns
        -------
        None.

        """
        def map_study_id(row):
            return self.get_study_id(row.hero_id, row.species, row.sex)
        
        df = pd.read_csv(data_path)
        df['hero_id'] = hero_id
        df['study_id'] = df.apply(map_study_id, axis=1)
        conn = self._connect()
        df.to_sql('data', con=conn, if_exists='append', index=False)

    def calc_rat_BW(self, time, strain, sex):
        """Calculate the rat body weight in kg using linear interpolation with age in days""" 
        rat_growth = self.rat_growth.copy()
        
        rat_growth_data = rat_growth.loc[(rat_growth.strain == strain) & 
                                     (rat_growth.sex == sex), ['age', 'BW_mean']].astype(float)
        
        #calc_age = sci.interp1d(rat_growth_data.BW_mean.values, rat_growth_data.age_hrs.values, fill_value=(rat_growth_data.age_hrs.values[0], rat_growth_data.age_hrs.values[-1]))
        calc_BW = sci.interp1d(rat_growth_data.age.values, rat_growth_data.BW_mean.values, bounds_error=False, fill_value=(rat_growth_data.BW_mean.values[0], rat_growth_data.BW_mean.values[-1]))
        return calc_BW(time)/1000
    def calc_rat_age(self, BW, strain, sex):
        rat_growth = self.rat_growth.copy()
        rat_growth_data = rat_growth.loc[(rat_growth.strain == strain) & 
                                    (rat_growth.sex == sex), ['age', 'BW_mean']].astype(float)
        calc_age = sci.interp1d(rat_growth_data.BW_mean.values, rat_growth_data.age.values, bounds_error=False, fill_value=(rat_growth_data.age.values[0], rat_growth_data.age.values[-1]))
        return calc_age(BW)

    def _prep_data(self, raw_data: pd.DataFrame):
        def _time_cor(row):
            """correct the time to be in days"""
            if (row.time_units == 'hour') or (row.time_units =='hours'):
                return row.time/24.
            elif (row.time_units == 'week') or (row.time_units == 'weeks'):
                return row.time*7
            elif (row.time_units == 'day') or (row.time_units == 'days'):
                return row.time
            elif (row.time_units == 'minute') or (row.time_units == 'minutes'):
                return row.time/60./24.
            else:
                raise UnitsException("Error: Time conversion for %s not defined for hero_id: %s"%(row.time_units, row.hero_id))
        def _dose_cor(row):
            """Correct the dose to be in mg/kg"""
            if row.dose_units in ['umol/kg']:
                MW = self.MW[row.pfas_abbrev]
                return row.dose*MW/1000.
            elif row.dose_units in ['mg/kg']:
                return row.dose
            else:
                raise UnitsException("Error: dose conversion for %s not defined for hero_id: %s"%(row.dose_units, row.hero_id))


        def _conc_cor(row, conc_col):
            """Convert concentrations to mg/L
            conc_col: "conc_mean" for concentration conversion, "conc_sd" for standard deviation conversion
            
            """

            if row[conc_col] > 0:
                if row.conc_units in ['ng/ml', 'ug/L']:
                    return row[conc_col]/1000.
                elif row.conc_units in ['nmol/ml', 'umol/L']:
                    MW = self.MW[row.pfas_abbrev]
                    return row[conc_col]*MW/1000
                elif (row.conc_units in ['mg/L', 'ug/ml']):
                    return row[conc_col]
                #elif row.conc_units in ['perc_dose']:
                #    return (row[conc_col]/100)*row['BW_init']
                #elif row.conc_units in ['perc_dose/g-tissue']:
                #    return (row[conc_col]*row['BW_init']*row['dose_cor']*1000)
                else:
                    raise UnitsException("Error: Concentration conversion for %s not defined for hero_id: %s"%(row.conc_units, row.hero_id))
            else:
                return row[conc_col]
        
        def _age_cor(row):
            if row.initial_age > 0 and not np.isnan(row.initial_age):
                # If there's a reported initial age, use it
                init_age = row.initial_age
                age_units = row.initial_age_units
                if age_units in ['months', 'month']:
                    return init_age*30.4374 # Convert months-->days
                elif age_units in ['weeks', 'week']:
                    return init_age*7 # Convert weeks-->days
                elif age_units in ['hour', 'hours']:
                    return init_age/24. # Convert hours-->days
                elif age_units in ['day', 'days']:
                    return init_age
                else:
                    raise UnitsException('Age units (%s) are not right for hero_id: %s' % (age_units, row.hero_id))
            
            if not np.isnan(row.initial_weight):
                # If initial weight is given, get the corresponding age (days) and follow the growth curve
                init_weight = row.initial_weight
                #init_age = calc_age(init_weight)
                return self.calc_rat_age(init_weight, row.strain, row.sex)
            else:
                if row.sex == 'Male':
                    return self.calc_rat_age(0.5, row.strain, row.sex)
                elif row.sex == 'Female':
                    return self.calc_rat_age(0.3, row.strain, row.sex)
                
        def _BW_cor(row):
            """Calculate BW in kg for given time point based on species/sex/strain"""
            if row.BW > 0 and not np.isnan(row.BW):
                time_course = 'Indiv. Reported'
                # There's a reported body weight so just need to convert to kg
                if row.BW_units in ['g', 'gram', 'grams']:
                    #cur_age = self.calc_rat_age(row.BW, row.strain, row.sex)
                    #init_age = cur_age - row.time_cor
                    #row['BW_cor'] = row.BW/1000.
                    #row['BW_init'] = self.calc_rat_BW(init_age, row.strain, row.sex)
                    row['time_course'] = time_course
                    row['BW_cor'] = row.BW/1000.

                    return row
                elif row.BW_units in ['kg', 'Kg', 'kilograms', 'kilogram', 'GD']:
                    #cur_age = self.calc_rat_age(row.BW*1000, row.strain, row.sex)
                    #init_age = cur_age - row.time_cor
                    #row['BW_cor'] = row.BW
                    #row['BW_init'] = self.calc_rat_BW(init_age, row.strain, row.sex)
                    row['time_course'] = time_course
                    row['BW_cor'] = row.BW
                    return row
                else:
                    raise UnitsException("Error: BW conversion for %s not defined for hero_id: %s"%(row.BW_units, row.hero_id))
            #rat_growth = self.rat_growth.copy()
            #rat_growth_data = rat_growth.loc[(rat_growth.strain == row.strain) & 
            #                         (rat_growth.sex == row.sex), ['age', 'BW_mean']].astype(float)
            if row.BW < 0 and not self.correct_BW:
                time_course = 'No Growth'
                tmp_BW = row.initial_weight
                tmp_BW_units = row.initial_weight_units
                if tmp_BW_units in ['g', 'gram', 'grams']:
                    row['time_course'] = time_course
                    row['BW_cor'] = tmp_BW/1000.
                    return row
                elif tmp_BW_units in ['kg', 'Kg', 'kilograms', 'kilogram']:
                    row['time_course'] = time_course
                    row['BW_cor'] = tmp_BW
                    return row
                else:
                    raise UnitsException("Error: BW conversion for %s not defined for hero_id: %s"%(tmp_BW_units, row.hero_id))

            #calc_age = sci.interp1d(rat_growth_data.BW_mean.values, rat_growth_data.age.values, bounds_error=False, fill_value=(rat_growth_data.age.values[0], rat_growth_data.age.values[-1]))
            if not np.isnan(row.BW_init):
            #if not row.BW_init.isnull():
                init_weight = row.BW_init
                init_age = self.calc_rat_age(init_weight, row.strain, row.sex)
            elif not np.isnan(row.initial_weight):
            #elif row.initial_weight is not None:
                # If initial weight is given, get the corresponding age (days) and follow the growth curve
                init_weight = row.initial_weight
                #init_age = calc_age(init_weight)
                init_age = self.calc_rat_age(init_weight, row.strain, row.sex)
            elif not np.isnan(row.initial_age):
            #elif not row.initial_age.isnull():
                # Weight is not given so convert initial age to weight
                init_age = row.initial_age
                age_units = row.initial_age_units
                if age_units in ['months', 'month']:
                    init_age = init_age*30.4374 # Convert months-->days
                elif age_units in ['weeks', 'week']:
                    init_age = init_age*7 # Convert weeks-->days
                elif age_units in ['hour', 'hours']:
                    init_age = init_age/24. # Convert hours-->days
                else:
                    raise UnitsException('Age units (%s) are not right for hero_id: %s' % (age_units, row.hero_id))
            else:
                print('No age/weight info, returning sex-specific adult weight for %s' % row.hero_id)
                time_course = 'Default'
                if row.sex == 'Male':
                    row['time_course'] = time_course
                    row['BW_cor'] = 0.5
                    return row
                elif row.sex == 'Female':
                    row['time_course'] = time_course
                    row['BW_cor'] = 0.3
                    return row
            time_course = row.time_cor + init_age
            #row['BW_cor'] = self.calc_rat_BW(time_course, row.strain, row.sex)
            #row['BW_init'] = self.calc_rat_BW(init_age, row.strain, row.sex)
            row['time_course'] = time_course
            row['BW_cor'] = self.calc_rat_BW(time_course, row.strain, row.sex)
            return row

        def _BW_init_cor(row):
            """Calculate BW in kg for given time point based on species/sex/strain"""
            if row.BW_init > 0 and not np.isnan(row.BW_init):
                # There's a reported body weight so just need to convert to kg
                if row.BW_init_units in ['g', 'gram', 'grams']:
                    #cur_age = self.calc_rat_age(row.BW, row.strain, row.sex)
                    #init_age = cur_age - row.time_cor
                    #row['BW_cor'] = row.BW/1000.
                    #row['BW_init'] = self.calc_rat_BW(init_age, row.strain, row.sex)
                    return row.BW_init/1000.
                elif row.BW_init_units in ['kg', 'Kg', 'kilograms', 'kilogram', 'GD']:
                    #cur_age = self.calc_rat_age(row.BW*1000, row.strain, row.sex)
                    #init_age = cur_age - row.time_cor
                    #row['BW_cor'] = row.BW
                    #row['BW_init'] = self.calc_rat_BW(init_age, row.strain, row.sex)
                    return row.BW_init
                else:
                    raise UnitsException("Error: BW conversion for %s not defined for hero_id: %s"%(row.BW_init_units, row.hero_id))

        def _BW_cor_no_growth(row):
            if row.BW > 0 and not np.isnan(row.BW):
                # There's a reported body weight so just need to convert to kg
                if row.BW_units in ['g', 'gram', 'grams']:
                    #cur_age = self.calc_rat_age(row.BW, row.strain, row.sex)
                    #init_age = cur_age - row.time_cor
                    #row['BW_cor'] = row.BW/1000.
                    #row['BW_init'] = self.calc_rat_BW(init_age, row.strain, row.sex)
                    return row.BW/1000.
                elif row.BW_units in ['kg', 'Kg', 'kilograms', 'kilogram', 'GD']:
                    #cur_age = self.calc_rat_age(row.BW*1000, row.strain, row.sex)
                    #init_age = cur_age - row.time_cor
                    #row['BW_cor'] = row.BW
                    #row['BW_init'] = self.calc_rat_BW(init_age, row.strain, row.sex)
                    return row.BW
                else:
                    raise UnitsException("Error: BW conversion for %s not defined for hero_id: %s"%(row.BW_units, row.hero_id))
            else:
                if row.BW > 0 and not np.isnan(row.BW):
                    # There's a reported body weight so just need to convert to kg
                    if row.BW_units in ['g', 'gram', 'grams']:
                        return row.BW/1000.
                    elif row.BW_units in ['kg', 'Kg', 'kilograms', 'kilogram', 'GD']:
                        return row.BW
                    else:
                        raise UnitsException("Error: BW conversion for %s not defined for hero_id: %s"%(row.BW_units, row.hero_id))
                elif not np.isnan(row.BW_init):
                    # No reported BW, but we have an initial weight for the animal
                    if row.BW_units in ['g', 'gram', 'grams']:
                        return row.BW_init/1000.
                    elif row.BW_units in ['kg', 'Kg', 'kilograms', 'kilogram', 'GD']:
                        return row.BW_init
                elif not np.isnan(row.initial_weight):
                #elif row.initial_weight is not None:
                    # No animal-specific weights, but we have study level summary weights
                    if row.initial_weight_units in ['g', 'gram', 'grams']:
                        return row.initial_weight/1000.
                    elif row.initial_weight_units in ['kg', 'Kg', 'kilograms', 'kilogram', 'GD']:
                        return row.initial_weight
                else:
                    # No animal body weights at all
                    print("No BW info for %s in %s, using defaults" % (row.species, row.hero_id))

                    if row.species == 'mouse':
                        if row.sex == 'Male':                        
                            return 0.031 # Average initial mass of Tatum-Gibbs male mice
                        else:
                            return 0.029 # Average initial mass of Tatum-Gibbs female mice
                    elif row.species == 'primate':
                        if row.sex == 'Male':
                            return 2.6 # M vs. F mass for Cynomolgus reported by Chengelis
                        else:
                            return 2.5


        
        def _sd_cor(row):
            if not row.standard_error:
                # Nothing to correct
                return row.conc_sd_cor
            else:
                # Convert standard error to standard deviation
                return row.conc_sd_cor*np.sqrt(row.N_animals)

        df = raw_data.copy()
        
        df['animal_id'] = df['animal_id'].fillna(-1)
        df['BW'] = df['BW'].fillna(value=np.nan)
        df['BW_init'] = df['BW_init'].fillna(value=np.nan)
        print('Hero ID available: ', df.hero_id.unique())

        # Filter out skip hero
        if self.skip_hero:
            df = df[~df.hero_id.isin(self.skip_hero)]
        # Get single hero_id
        if self.hero_only:
            df = df[df.hero_id == self.hero_only]
        
        df['route'] = df['route'].str.lower()
        # Get single route
        if self.route_only:
            df = df[df.route == self.route_only]
            
        # Filter any data
        if self.fit_only_blood:
            df = df[df.matrix.isin(['blood', 'plasma', 'serum'])]
        
        # Remote blood data from Iwabuchi and only use serum
        df = df[~((df.hero_id==3859701) & (df.matrix=='blood'))] # this is the only study with both blood and serum measurements. Using serum to keep it simple.


        if self.sex != 'combined':
            df = df[df.sex==self.sex]
        
        df = df[df.species==self.species]
        if df.empty:
            raise DataException("Error: %s is not available for %s %s" % (self.chemical, self.sex, self.species))
        
        df = df[~df.matrix.isin(self.skip_matrix)]

        
        if self.remove_single_times:
            df['n_times'] = df.groupby(['hero_id', 'species', 'dose', 'matrix', 'route'])['time'].transform('nunique')
            df = df[df.n_times>1] # Only keep datasets that have more than one time point

        
        #df = df[df.hero_id.isin([3858670, 3861408])]
        #df = df[(df.idx1 == 0) & (df.sex=='Male')]
        
        # idx1: Indexing for each study
        # idx2: Indexing for each dataset inside a study (reset to 0 for each new study)
        # idx3: Index every unique dataset 0 to n-1 total datasets
        
        
        df['idx1'] = df.groupby(['hero_id'], sort=False).ngroup()
        df['idx2'] = 0
        df['idx3'] = df.groupby(['hero_id', 'matrix', 'route', 'dose'], sort=False).ngroup()
        #df['idx3'] = df.groupby(['hero_id', 'matrix', 'route', 'dose', 'animal_id'], sort=False).ngroup()
        
        
        for h in df.hero_id.unique():
            tmp_df = df[df.hero_id==h]
            df.loc[tmp_df.index, 'idx2'] = tmp_df.groupby(['matrix', 'route', 'dose'], sort=False).ngroup()

        df['time_cor'] = df.apply(_time_cor, axis=1) # Make sure times are in days
        df['dose_cor'] = df.apply(_dose_cor, axis=1) # Make sure doses are in mg/kg
        
        # Handle changing body weight
        #df= df.apply(_BW_cor, axis=1)
        

        if self.species == 'rat':
            df = df.apply(_BW_cor, axis=1)
            df['BW_cor'] = df['BW_cor'].astype(float)
        
        elif self.species in ['mouse', 'monkey', 'primate']:
            df['BW_cor'] = df.apply(_BW_cor_no_growth, axis=1).astype(float)
        
        
        
        df['BW_init']= df.apply(_BW_init_cor, axis=1)
        #df['init_age'] = df.apply(_age_cor, axis=1)
        #df['time_age'] = df['time_cor'] + df['init_age']
        
        #df['BW_init'] = df.loc[df.groupby(['animal_id', 'hero_id', 'route'])['time_cor'].transform('idxmin'), 'BW_cor'].values
        
        # Handle BW_init when we have values
        tmp_BW = df.loc[df.groupby(['animal_id', 'hero_id', 'route'])['time_cor'].transform('idxmin'), 'BW_cor'].copy()
        tmp_BW = tmp_BW.reset_index(drop=True)
        tmp_BW.index = df.index # Match the indices for fillna
        df['BW_init'] = df['BW_init'].fillna(tmp_BW)

        df.loc[df.conc_units.isin(['perc_dose']), 'BW_cor'] = df.loc[df.conc_units.isin(['perc_dose']), 'BW_init']
        
       
  
        
        df[df.conc_mean.isin(['BLOQ'])] = -1 # Change any LoD values to -1
        df['conc_mean'] = df.conc_mean.astype(float)
        df['conc_sd'] = df.conc_sd.astype(float)
        
        df.loc[df.conc_sd < 0, 'conc_sd'] = 0
        

        df['conc_mean_cor'] = df.apply(_conc_cor, axis=1, args=('conc_mean',)) # Convert concs to mg/L
        df['conc_sd_cor'] = df.apply(_conc_cor, axis=1, args=('conc_sd',)) # Convert sd to mg/L
        
        df['conc_sd_cor'] = df.apply(_sd_cor, axis=1) # Make sure standard error --> standard deviation correction is done
        df.loc[df.conc_sd_cor > df.conc_mean_cor, 'conc_sd_cor'] = 0 # SD > mean almost always because of digitize errors

        df.loc[df.N_animals == 'NR', 'N_animals'] = 1 # Handle NR animals

        df['N_animals'] = df['N_animals'].astype(int)
        df['Dss'] = 0

        # Tatum-Gibbs correction
        #-----------------------
        # Backgound levels of PFNA were measured in control animal serum. To account for this, either
        # a "background" concentration is defined and subtracted from the concentration reported in the study.
        # From Tatum-Gibbs, these concentrations were 0.06 ug/ml (male rat), 5.55 ng/ml (male mouse), and 8.99 ng/ml (female mouse)
        df.loc[(df.hero_id == 2919268) & (df.species == 'rat') & (df.sex == 'Male'), 'conc_mean_cor'] -= 0.06
        df.loc[(df.hero_id == 2919268) & (df.species == 'mouse') & (df.sex == 'Male'), 'conc_mean_cor'] -= 0.00555
        df.loc[(df.hero_id == 2919268) & (df.species == 'mouse') & (df.sex == 'Female'), 'conc_mean_cor'] -= 0.00899

        # For male rats, assume a 0.06 ug/ml steady-state concentration (Cpss) and assume same dose (D) applied to control male and female rats through drinking water
        # Model derived clearance for PFNA in male rats is XXX (CL)
        # Cpss = D/CL --> background dose to female rats is XXX mg/kg/d
        # This steady state dose is accounted for in the PK model
        df.loc[(df.hero_id == 2919268) & (df.species == 'rat') & (df.sex == 'Female'), 'Dss'] = 0.000218 # mg/kg/d background dose

        #-----------------------


        # Handle possible censored data (not used)
        if 'lod' in df.columns:
            df['lod'] = df['lod'].fillna(-1)
        else:
            df['lod'] = -1
        if not self.censor:
            df = df[(df.conc_mean_cor > 0) & (df.conc_mean_cor > df.lod)]
            df['censored'] = 0
        else:
            df['lod'] = df.lod.astype(float)
            df['lod'] = df.apply(_conc_cor, axis=1, args=('lod',)) # Convert LoD to mg/L
            df['censored'] = (df['conc_mean_cor'] < df['lod']).astype(int)
            df.loc[df.conc_mean_cor < df.lod, 'conc_mean_cor'] = df.loc[df.conc_mean_cor < df.lod, 'lod']
            df = df[df.conc_mean_cor > 0]
        
        #df = df[~df.conc_mean.isin(['BLOQ'])]
        #df = df[df.conc_mean > 0] # Filter missing data for now
        
        

        df['route_idx'] = (~df.route.isin(['iv', 'iv-repeat'])).astype(int) # 0: IV, IV-repeat, 1: Everything else
        df['sex_idx'] = (df.sex=='Female').astype(int) # 1: Female, 0: Male
        
        # Scale the data
        if self.scale:
            df['dataset_max'] = df.groupby(['idx3'])['conc_mean_cor'].transform('max')
            df['conc_scaled'] = df['conc_mean_cor']/df['dataset_max']
            df['sd_scaled'] = df['conc_sd_cor']/df['dataset_max']
            
            lnsd2 = np.log(1+(df['sd_scaled']**2)/(df['conc_scaled']**2))
            df['lnconc_sd'] = np.sqrt(lnsd2)
            df['lnconc_mean'] = np.log(df['conc_scaled']) - 0.5*lnsd2
        else:
            lnsd2 = np.log(1+(df['conc_sd_cor']**2)/(df['conc_mean_cor']**2))
            df['lnconc_sd'] = np.sqrt(lnsd2)
            df['lnconc_mean'] = np.log(df['conc_mean_cor']) - 0.5*lnsd2

        df['cv'] = df['conc_sd_cor']/df['conc_mean_cor']
        df['aidx'] = df['animal_id'].replace(-1, np.nan).notna()
        df['dose_mg'] = df.dose_cor*df.BW_init # mg applied dose
        df['dataset_str'] = df.hero_id.astype(str) + '-' + np.round(df.dose_cor,2).astype(str) + ' mg/kg-' + df.route # Unique identifier for each dataset (hero/dose/route)

        #df['lnconc_mean_scaled'] = df['lnconc_mean']/df['dataset_lnmax']
        #df['lnconc_sd_scaled'] = df['lnconc_sd']/df['dataset_lnmax']



        
        print('dropping duplicated column')
        df = df.loc[:,~df.columns.duplicated()]
        df = df.sort_values(by=['idx3', 'time_cor'])
        
        # Reset index
        df.reset_index(drop=True, inplace=True)
        #self.data = df.copy()
        print('Hero ID used: ', df.hero_id.unique())
    
        return df
    
    def get_processed_data(self, chemical, sex, species):
        """Query the literautre extracted, raw data, and add the necessary fields through self._prep_data

        Parameters
        ----------
        chemical : str
            Chemical (PFAS abbrev.) to query from database
        sex : str
            sex of species to query. Use 'combined' for both M/F
        species : _type_
            species to query

        Returns
        -------
        pandas.DataFrame
            Pandas dataframe containing the processed data
        """
        self.chemical = chemical
        self.sex = sex
        self.species = species
        self.raw_data = self.get_data(self.chemical, study=True)
        return self._prep_data(self.raw_data)
