# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:25:12 2021

@author: tzurlind
"""
import numpy as np

def register_data(PFAS):
    #======================
    # Gannon et al. 2850314
    #======================
    print('Gannon et al. 2850314')
    hero_id = 2850314
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=4, age=6, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Gannon')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=4, age=6, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Gannon')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Male', n=3, age=6, age_units='weeks', weight=30, weight_units='grams', author='Gannon')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Female', n=3, age=6, age_units='weeks', weight=30, weight_units='grams', author='Gannon')
    
    PFAS.register_data('../extracted_data/Gannon_2850314.csv', hero_id=hero_id)
    
    #======================
    # Ohmori et al. 3858670
    #======================
    print('Ohmori et al. 3858670')
    hero_id = 3858670
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=24, age_units='months', weight=np.nan, weight_units='Not Reported', author='Ohmori')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=3, age=24, age_units='months', weight=np.nan, weight_units='Not Reported', author='Ohmori')
    
    PFAS.register_data('../extracted_data/Ohmori_3858670.csv', hero_id=hero_id)
       
    
    #============
    # Kim 5063958
    #============
    print('Kim 5063958')
    hero_id = 5063958
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=5, age=7.5, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Kim')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=5, age=7.5, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Kim')
    
    PFAS.register_data('../extracted_data/Kim_5063958.csv', hero_id=hero_id)
    
    #===================
    # Dzierlenga 5916078
    #===================
    print('Dzierlenga 5916078')
    hero_id = 5916078
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=8, age_units='weeks', weight=np.nan, weight_units='Indiv. Reported', author='Dzierlenga')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=3, age=8, age_units='weeks', weight=np.nan, weight_units='Indiv. Reported', author='Dzierlenga')
    PFAS.register_data('../extracted_data/Dzierlenga_5916078.csv', hero_id=hero_id)
    
    
    #===================
    # Sundstrom 1289834
    #===================
    print('Sundstrom 1289834')
    hero_id = 1289834
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Sundstrom')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=3, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Sundstrom')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Male', n=4, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Sundstrom')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Female', n=4, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Sundstrom')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Male', n=3, age=np.nan, age_units='Adult', weight=np.nan, weight_units='Indiv. Reported', author='Sundstrom')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Female', n=3, age=np.nan, age_units='Adult', weight=np.nan, weight_units='Indiv. Reported', author='Sundstrom')
    
    PFAS.register_data('../extracted_data/Sundstrom_1289834.csv', hero_id=hero_id)
    
    #===================
    # Kim 3749289
    #===================
    print('Kim 3749289')
    hero_id = 3749289
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=5, age=10, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Kim')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=5, age=10, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Kim')

    PFAS.register_data('../extracted_data/Kim_3749289.csv', hero_id=hero_id)
    
    #===================
    # Kim 4239569
    #===================
    print('Kim 4239569')
    hero_id = 4239569
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=5, age=10, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Kim')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=5, age=10, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Kim')
    
    PFAS.register_data('../extracted_data/Kim_4239569.csv', hero_id=hero_id)
    
    #===================
    # Iwabuchi 3859701
    #===================
    print('Iwabuchi 3859701')
    hero_id = 3859701
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=6, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Iwabuchi')
    PFAS.register_data('../extracted_data/Iwabuchi_3859701.csv', hero_id=hero_id)
    
    #====================
    # Tatum-Gibbs 2919268
    #====================
    print('Tatum-Gibbs 2919268')
    hero_id = 2919268
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=1, age=11, age_units='weeks', weight=np.nan, weight_units='Indiv. Reported', author='Tatum-Gibbs')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=1, age=11, age_units='weeks', weight=np.nan, weight_units='Indiv. Reported', author='Tatum-Gibbs')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Male', n=1, age=9, age_units='weeks', weight=np.nan, weight_units='Indiv. Reported', author='Tatum-Gibbs')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Female', n=1, age=9, age_units='weeks', weight=np.nan, weight_units='Indiv. Reported', author='Tatum-Gibbs')
    
    PFAS.register_data('../extracted_data/Tatum-Gibbs_2919268.csv', hero_id=hero_id)
    
    #==============
    # Huang 5387170
    #==============
    print('Huang 5387170')
    hero_id = 5387170
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=1, age=8, age_units='weeks', weight=np.nan, weight_units='Indiv. Reported', author='Huang')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=1, age=8, age_units='weeks', weight=np.nan, weight_units='Indiv. Reported', author='Huang')
    
    PFAS.register_data('../extracted_data/Huang_5387170.csv', hero_id=hero_id)
    
    #==============
    # Olsen 1326734
    #==============
    print('Olsen 1326734')
    hero_id = 1326734
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Olsen')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=3, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Olsen')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Male', n=1, age=2.75, age_units='years', weight=np.nan, weight_units='Not Reported', author='Olsen')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Female', n=1, age=2.75, age_units='years', weight=np.nan, weight_units='Not Reported', author='Olsen')
    
    PFAS.register_data('../extracted_data/Olsen_1326734.csv', hero_id=hero_id)
    
    #==============
    # Chang 2325359
    #==============
    print('Chang 2325359')
    hero_id = 2325359
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=9, age_units='weeks', weight=225, weight_units='grams', author='Chang')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=3, age=9, age_units='weeks', weight=225, weight_units='grams', author='Chang')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Male', n=3, age=9, age_units='weeks', weight=30, weight_units='grams', author='Chang')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Female', n=3, age=9, age_units='weeks', weight=30, weight_units='grams', author='Chang')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Male', n=1, age=2.75, age_units='years', weight=5, weight_units='kilograms', author='Chang')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Female', n=1, age=2.75, age_units='years', weight=4, weight_units='kilograms', author='Chang')
    
    PFAS.register_data('../extracted_data/Chang_2325359.csv', hero_id=hero_id)
    
    
    #==============
    # Daikin 6822782
    #==============
    print('Daikin 6822782')
    hero_id = 6822782
    # Register study
    #---------------
    
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Female', n=1, age=np.nan, age_units='Not Reported', weight=np.nan, weight_units='Indiv. Reported', author='Daikin')
    PFAS.register_data('../extracted_data/Daikin_6822782.csv', hero_id=hero_id)
    
    #==============
    # Chengelis 2850396
    #==============
    print('Chengelis 2850396')
    hero_id = 2850396
    # Register study
    #---------------
    
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=7, age_units='weeks', weight=238, weight_units='grams', author='Chengelis')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=3, age=7, age_units='weeks', weight=187, weight_units='grams', author='Chengelis')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Male', n=1, age=3, age_units='years', weight=2.6, weight_units='kilograms', author='Chengelis')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Female', n=1, age=3, age_units='years', weight=2.5, weight_units='kilograms', author='Chengelis')
    
    PFAS.register_data('../extracted_data/Chengelis_2850396.csv', hero_id=hero_id)
    
    #==============
    # Lau 6579272
    #==============
    print('Lau 6579272')
    hero_id = 6579272
    # Register study
    #---------------
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Male', n=1, age=10, age_units='weeks', weight=31, weight_units='grams', author='Lau')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Female', n=1, age=10, age_units='weeks', weight=29, weight_units='grams', author='Lau')
    
    PFAS.register_data('../extracted_data/Lau_6579272.csv', hero_id=hero_id)


    #==============
    # Chang 1289832
    #==============
    print('Chang 1289832')
    hero_id = 1289832
    # Register study
    #---------------
    
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=9, age_units='weeks', weight=300, weight_units='grams', author='Chang')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=3, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Chang')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Male', n=4, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Chang') 
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Female', n=4, age=9, age_units='weeks', weight=np.nan, weight_units='Not Reported', author='Chang') 
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Male', n=1, age=2.75, age_units='years', weight=np.nan, weight_units='Not Reported', author='Chang')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Female', n=1, age=2.75, age_units='years', weight=np.nan, weight_units='Not Reported', author='Chang')

    PFAS.register_data('../extracted_data/Chang_1289832.csv', hero_id=hero_id)


    #==============
    # Butenhoff 3749227
    #==============
    print('Butenhoff 3749227')
    hero_id = 3749227
    # Register study
    #---------------

    PFAS.register_study(hero_id=hero_id, species='primate', sex='Male', n=1, age=2.75, age_units='years', weight=5, weight_units='kilograms', author='Butenhoff')
    PFAS.register_study(hero_id=hero_id, species='primate', sex='Female', n=1, age=2.75, age_units='years', weight=4, weight_units='kilograms', author='Butenhoff') # Same monkeys as Chang 2325359

    PFAS.register_data('../extracted_data/Butenhoff_3749227.csv', hero_id=hero_id)

    #==============
    # Kudo 2990271
    #==============
    print('Kudo 2990271')
    hero_id = 2990271
    # Register study
    #---------------

    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=3, age=10, age_units='weeks', weight=280, weight_units='grams', author='Kudo')
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=3, age=10, age_units='weeks', weight=213, weight_units='grams', author='Kudo') 

    PFAS.register_data('../extracted_data/Kudo_2990271.csv', hero_id=hero_id)

    #==============
    # Lou 2919359
    #==============
    print('Lou 2919359')
    hero_id = 2919359
    # Register study
    #---------------

    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Male', n=3, age=75, age_units='days', weight=np.nan, weight_units='Not Reported', author='Lou')
    PFAS.register_study(hero_id=hero_id, species='mouse', sex='Female', n=3, age=75, age_units='days', weight=np.nan, weight_units='Not Reported', author='Lou')

    PFAS.register_data('../extracted_data/Lou_2919359.csv', hero_id=hero_id)

    #==============
    # Kemper 6302380
    #==============
    print('Kemper 6302380')
    hero_id = 6302380
    # Register study
    #---------------

    PFAS.register_study(hero_id=hero_id, species='rat', sex='Male', n=1, age=np.nan, age_units='Mature', weight=np.nan, weight_units='Indiv. Reported', author='Kemper') # From CR growth curves
    PFAS.register_study(hero_id=hero_id, species='rat', sex='Female', n=1, age=np.nan, age_units='Mature', weight=np.nan, weight_units='Indiv. Reported', author='Kemper') # From CR growth curves

    PFAS.register_data('../extracted_data/Kemper_6302380.csv', hero_id=hero_id)

    
    
    