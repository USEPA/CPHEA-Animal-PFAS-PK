# Possible modules for future ODE-based models
#import sunode
#import sunode.wrappers.as_aesara
#import sympy as sym
#from sunode.wrappers.as_aesara import solve_ivp

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import pandas as pd
import scipy.optimize as sco
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import scipy.interpolate as sci
import sklearn.metrics as skm
import seaborn as sns

import arviz as az
import arviz.labels as azl
import pymc as pm
import xarray as xr
import pymc.util as pm_util
import pytensor.tensor as pt
import pytensor.printing as pt_print
import os


def my_logp_lognormal(x, lnsummary_conc, sigma_summary, Ni_summary, sd_summary):
    """logp calculation for lognormally distributed data

    Parameters
    ----------
    x : _type_
        _description_
    lnsummary_conc : _type_
        _description_
    sigma_summary : _type_
        _description_
    Ni_summary : _type_
        _description_
    sd_summary : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    # X is regular observed conc
    
    # Log-transform summary data
    lnsd2_summary = pm.math.log(1+(sd_summary**2)/(x**2)) # This is lnsd_summary**2 (s_{L})^2
    lny_summary = np.log(x) - 0.5*lnsd2_summary
    
    term1 = (Ni_summary*lny_summary)
    term2 = (Ni_summary/2.)*pm.math.log(sigma_summary**2)
    term3 = ((Ni_summary - 1)*lnsd2_summary)/(2*sigma_summary**2)
    term4 = (Ni_summary*(lny_summary - lnsummary_conc)**2)/(2*sigma_summary**2)
    #LL = (N/2)*pm.math.log(2*np.pi) - pm.math.sum((term1 + term2 + term3 + term4))
    logp = -(Ni_summary/2.)*pm.math.log(2*np.pi) - (term1 + term2 + term3 + term4)
    return logp
def my_random_lognormal(mu, sd, Ni_summary, sd_summary, rng=None, size=None):
    """random sampler for log-normal distribution

    Parameters
    ----------
    mu : float
        mean of log-normal distribution
    sd : float
        standard deviation of log-normal distribution
    Ni_summary : _type_
        Placeholder variable, not used
    lnsd2_summary : _type_
        Placeholder variable, not used
    rng : np.rng, optional
        numpy random number generator used by PyMC, by default None
    size : int, optional
        Size of sample, by default None

    Returns
    -------
    _type_
        _description_
    """
    
    #return rng.lognormal(mean=np.exp(mu), sigma=sd, size=size)
    return np.exp(rng.normal(loc=mu, scale=sd, size=size))

def my_logp_normal(x, summary_conc, cv_summary, Ni_summary, sd_summary):
    """logp for normally distributed data

    Parameters
    ----------
    x : _type_
        _description_
    summary_conc : _type_
        _description_
    cv_summary : _type_
        _description_
    Ni_summary : _type_
        _description_
    sd_summary : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    sigma_summary = cv_summary*summary_conc
    term1 = (Ni_summary/2.)*pm.math.log(sigma_summary**2)
    term2 = ((Ni_summary -1)*sd_summary**2)/(2*sigma_summary**2)
    term3 = (Ni_summary*(x - summary_conc)**2)/(2*sigma_summary**2)
    logp = -(Ni_summary/2.)*pm.math.log(2*np.pi) - (term1 + term2 + term3)
    return logp
def my_random_normal(mu, sd, Ni_summary, lnsd2_summary, rng=None, size=None):
    """random sampler for normal distribution

    Parameters
    ----------
    mu : float
        mean of normal distribution
    sd : float
        standard deviation of normal distribution
    Ni_summary : _type_
        Placeholder variable, not used
    lnsd2_summary : _type_
        Placeholder variable, not used
    rng : np.rng, optional
        numpy random number generator used by PyMC, by default None
    size : int, optional
        Size of sample, by default None

    Returns
    -------
    _type_
        _description_
    """
    return rng.normal(loc=mu, scale=sd, size=size)

class PyPKMC:
    def __init__(self, data, chem_label='pfas_abbrev', sex_label = 'sex', species_label='species', fit_fa = True, default_fa=1.,
                time_label='time', y_obs_label = 'conc_mean', sd_obs_label = 'sd_obs', 
                route_label = 'route_idx', dose_label='dose_mg', BW_label='BW', 
                study_label='study', dataset_label='dataset', indiv_label='aidx', N_label='N_animals', Dss_label = 'Dss',
                CLC_prior = {'mu': -2.65, 'sd': 2.68}, Vdss_prior = {'mu': -1.02, 'sd': 1}, pop_SD_prior = 0.5, abs_SD_prior=0.2, Rkcd_SD_prior = 0.5):
        """PyPKMC class to run the MCMC sampling. All keyword arguments with *_label in __init__ represent column names
           in data containing the specified information

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe containing the necessary observed data at each time point
        chem_label : str, optional
            data column containing the chemical name, by default 'pfas_abbrev'
        sex_label : str, optional
            data column containing the sex, by default 'sex'
        species_label : str, optional
            data column containing the species, by default 'species'
        fit_fa : bool, optional
            Boolean to manually turn off fitting fa (fit_fa=False). If fit_fa=True, check to see if both gavagea and IV routes available before proceeding, by default True
        default_fa : float, optional
            Default fraction absoprbed if we are not fitting fa. Default is 100%
        time_label : str, optional
            data column containing the time for fitting, by default 'time'
        y_obs_label : str, optional
            data column containing the observed mean concentration for fitting, by default 'conc_mean'
        sd_obs_label : str, optional
            data column containing the observed standard deviation for fitting, by default 'sd_obs'
        route_label : str, optional
            data column containing the route of exposure, by default 'route_idx'
        dose_label : str, optional
            data column containing the applies dose (in mg), by default 'dose_mg'
        BW_label : str, optional
            data column containing the body weights for each time point, by default 'BW'
        study_label : str, optional
            data column containing the study-level label for each observed concentration, by default 'study'
        dataset_label : str, optional
            data column containing the dataset-level label for each observed concentration, by default 'dataset'
        indiv_label : str, optional
            data column to characterize the obsrrved concentrationa as either 1 (for indiviudal animal concentration) or 0 (summary level concentration), by default 'aidx'
        N_label : str, optional
            data column with the number of animals contributing to eahc observed concentration, by default 'N_animals'
        Dss_label: str, optional
            label for column with steady state background dose, by default 'D_ss'
        CLC_prior : dict, optional
            Mean (mu) and standard deviation (sd) prior (log-transformed) for clearance (L/d/kg), by default {'mu': -2.65, 'sd': 2.68}
            log(mu) = -2.68 --> mu = 0.07 L/d/kg, sd is already log-transformed and used. log(CLC) ~ N(log(0.07), 2.68)
        Vdss_prior : dict, optional
            Mean (mu) and standard deviation (sd) prior (log-transformed) for volume of distribution (L/kg), by default {'mu': -1.02, 'sd': 1}
            log(mu) = -1.02 --> mu = 0.36 L/kg, sd is already log-transformed and used. log(Vdss) ~ N(log(0.36), 1)
        pop_SD_prior : float, optional
            Prior for population HalfNormal distribution where sigma_param ~ HalfNormal(pop_SD_prior)
        abs_SD_prior : float, optional
            Prior for populaiton HalfNormal distribution for k_abs sigma
        Rkcd_SD_prior : float, optional
            Priot for population HalfNormal distributions for k_cd and R (where autocorrelation can happen)
        """
        
        
        self.time_label = time_label
        self.y_obs_label = y_obs_label
        self.sd_obs_label = sd_obs_label
        self.route_label = route_label
        self.dose_label = dose_label
        self.BW_label = BW_label
        self.study_label = study_label
        self.dataset_label = dataset_label
        self.indiv_label = indiv_label
        self.N_label = N_label
        self.Dss_label = Dss_label
        self.pop_SD_prior = pop_SD_prior
        self.abs_SD_prior = abs_SD_prior
        self.Rkcd_SD_prior = Rkcd_SD_prior
        self.default_fa = default_fa

        self.chemical = data[chem_label].unique()[0]
        self.sex = data[sex_label].unique()[0]
        self.species = data[species_label].unique()[0]
        self.units = {'CLC': 'ml/kg/d', 'Vdss': 'L/kg', 'halft': 'd', 'halft_beta': 'd'}
        self.description = {'CLC': 'Clearance', 'Vdss': 'Volume of distribution', 'halft': 'Half-life', 'halft_beta': 'Term. Half-life'}

        # Prior (Define default priors here)
        def calc_log_dist(dist_dict):
            lnsd2 = np.log(1+(dist_dict['sd']**2)/(dist_dict['mu']**2))
            lnsd = np.sqrt(lnsd2)
            lny = np.log(dist_dict['mu']) - 0.5*lnsd2
            return {'mu': lny, 'sd': lnsd}
        #self.CLC_prior = calc_log_dist(CLC_prior)
        #self.Vdss_prior = calc_log_dist(Vdss_prior)

        self.CLC_prior = CLC_prior
        self.Vdss_prior = Vdss_prior

        # MCMC attributes
        self.model = None
        self._trace = None

        # Factor the dataframe
        study_idx, study_codes = pd.factorize(data[study_label])
        dataset_idx, dataset_codes = pd.factorize(data[dataset_label])
        #time_idx, time_codes = pd.factorize(data[time_label])
        self.coords = {'study': study_codes, 'dataset': dataset_codes}

        self.study_idx, self.study_codes = study_idx, study_codes
        self.dataset_idx, self.dataset_codes = dataset_idx, dataset_codes
        self.gavage_codes = [x for x in self.dataset_codes if 'gavage' in x]

        # Define dataset codes that have iv and oral for same dose in study
        n_routes = data.groupby(['hero_id', 'dose'])['route'].nunique() 
        iv_oral = n_routes[n_routes>1].reset_index()
        self.iv_oral_codes = []
        for idx, row in iv_oral.iterrows():
            for route in ['iv', 'gavage']:
                self.iv_oral_codes.append("%s-%s mg/kg-%s"%(int(row.hero_id), np.round(row.dose,2), route))

        data['study_idx'] = study_idx
        data['dataset_idx'] = dataset_idx

        # Define dataset codes for highest and lowest gavage dose in a study
        gavage_only = data[data.route=='gavage'].copy()
        high_low = gavage_only.groupby('hero_id').agg({'dose': ['min', 'max']})
        high_low.columns =high_low.columns.droplevel()
        high_low = high_low.reset_index()
        high_low['diff'] = high_low['max'] - high_low['min']
        self.min_max_codes = []
        for idx, row in high_low.iterrows():
            #print('DIFFERENCE', row['diff'])
            if row['diff'] == 0:
                continue
            self.min_max_codes.append("%s-%s mg/kg-%s"%(int(row.hero_id), np.round(row['min'],2), 'gavage'))
            self.min_max_codes.append("%s-%s mg/kg-%s"%(int(row.hero_id), np.round(row['max'],2), 'gavage'))

        self.data = data.reset_index(drop=True)

        # Decide is fa is going to be a fitted param
        if fit_fa:
            self.fit_fa = (len(self.data[route_label].unique()) >= 2)
        else:
            self.fit_fa = False


    def _1cmpt_hierarchical(self, time, y_obs, sd_obs, route_idx, dose, dose_ss, BW, study_idx, dataset_idx, coords, indiv_bool, Ni):

        """1-compartment hierarchical model specification. All inputs are vectors of length = N concentrations"""

        #Ni: Number of animals per time point
        indiv_idx = np.where(indiv_bool)[0]
        summary_idx = np.where(~indiv_bool)[0]

        with pm.Model(coords=coords) as model:
            # use pm.Data(..., mutable=True) to specify variables that can be changed during posterior-predictive check
            y_indiv = y_obs.values[indiv_bool] # Observed data from individually reported data
            y_summary = y_obs.values[~indiv_bool] # Observed mean data from summary data
            sd_summary = sd_obs.values[~indiv_bool] # Observed sd data from summary data

            t = pm.Data("t", time, mutable=True)
            route = pm.Data("route", route_idx, mutable=True)
            D = pm.Data("D", dose, mutable=True)
            Dss = pm.Data("Dss", dose_ss, mutable=True)
            BW = pm.Data("BW", BW, mutable=True)
            sidx = pm.Data('sidx', study_idx, mutable=True)
            indiv_sidx = sidx[indiv_idx]
            summary_sidx = sidx[summary_idx]
            didx = pm.Data('didx', dataset_idx, mutable=True)

            # Population mean priors
            #-----------------------
            mu_lnCLC = pm.Normal('mu_lnCLC', self.CLC_prior['mu'], self.CLC_prior['sd'])
            mu_lnVd = pm.Normal('mu_lnVd', self.Vdss_prior['mu'], self.Vdss_prior['sd'])
            #mu_lnk_abs = pm.Normal('mu_lnk_abs', 2, 0.25, initval=3)
            #mu_lnk_abs = pm.Normal('mu_lnk_abs', np.log(49.2), 0.25, initval=3) # absorption t1/2 < 1 hours
            mu_lnk_abs = pm.Normal('mu_lnk_abs', np.log(81), 0.25, initval=3) # absorption t1/2 < 1 hours
            
            mu_CLC = pm.math.exp(mu_lnCLC)
            mu_Vd = pm.math.exp(mu_lnVd)
            mu_k_elim = pm.Deterministic('mu_k_elim', mu_CLC/mu_Vd)
            mu_k_abs = pm.Deterministic('mu_k_abs', pm.math.exp(mu_lnk_abs))

            #fa fit
            #================
            if self.fit_fa:
                #mu_fa = pm.Beta('mu_fa', 1,1)
                mu_fa = pm.Beta('mu_fa', 6, 1.5)
                kappaMinusTwo = pm.Gamma('kappaMinusTwo', 0.1, 0.1) # Krushke and Vanpaemel "Bayesian Estimation in Hierarchical Models"
                kappa = pm.Deterministic('kappa', kappaMinusTwo + 2) 
                fa = pm.Beta("fa", alpha=mu_fa*(kappa-2.0)+1.0, beta=(1.0 - mu_fa)*(kappa - 2.0) + 1, dims='dataset')
            else:
                #fa = pm.math.ones_like(didx)
                mu_fa = self.default_fa
                fa = pm.math.ones_like(np.unique(dataset_idx))*self.default_fa
            #==================
            

            # Population PK parameters
            pm.Deterministic('halft [pop]', pm.math.log(2)/mu_k_elim) # d
            pm.Deterministic('halft_beta [pop]', pm.math.log(2)/mu_k_elim) # d
            pm.Deterministic('CLC [pop]', pm.math.exp(mu_lnCLC)) # l/kg/d
            pm.Deterministic('Vdss [pop]', pm.math.exp(mu_lnVd)) # l/kg
            pm.Deterministic('abs_halft [pop]', pm.math.log(2)/mu_k_abs) # l/kg
            if self.fit_fa:
                pm.Deterministic('fa [pop]', mu_fa)

            
            mu_V = mu_Vd*BW

            mu_conc_iv = (D/mu_V)*pm.math.exp(-mu_k_elim*t)
            mu_conc_abs = (D/mu_V)*(mu_k_abs/(mu_k_abs-mu_k_elim))*(pm.math.exp(-mu_k_elim*t) - pm.math.exp(-mu_k_abs*t))
            
            mu_conc = pm.Deterministic('mu_conc', pt.switch(route < 0.5, mu_conc_iv, mu_conc_abs)) # iv == 0, non-iv == 1

            # Population standard-deviation priors
            #-------------------------------------
            SD = self.pop_SD_prior
            SD2 = 0.2 # 0.18
            # sigma_lnk_abs = pm.Exponential('sigma_lnk_abs', 1./CV)
            # sigma_lnVd = pm.Exponential('sigma_lnVd', 1./CV)
            # sigma_lnCLC = pm.Exponential('sigma_lnCLC', 1./CV)
            sigma_lnk_abs = pm.HalfNormal('sigma_lnk_abs', self.abs_SD_prior)
            sigma_lnVd = pm.HalfNormal('sigma_lnVd', SD)
            sigma_lnCLC = pm.HalfNormal('sigma_lnCLC', SD)

            pm.Deterministic('log_sigma_lnk_abs', pm.math.log(sigma_lnk_abs))
            pm.Deterministic('log_sigma_lnVd', pm.math.log(sigma_lnVd))
            pm.Deterministic('log_sigma_lnCLC', pm.math.log(sigma_lnCLC))

            # Population geometric standard deviation parameters where GSD = e^sigma
            # halft = f(CLC, Vd)
            pm.Deterministic('CLC [GSD]', pm.math.exp(sigma_lnCLC))
            pm.Deterministic('Vdss [GSD]', pm.math.exp(sigma_lnVd))
            pm.Deterministic('halft [GSD]', pm.math.exp(pm.math.sqrt(sigma_lnCLC**2 + sigma_lnVd**2)))
            
            # daatset-level sampling (reparameterization to help with sampling)
            #------------------------------------------------------------------
            lnk_abs_offset = pm.Normal('lnk_abs_offset', mu=0, sigma=1, dims='dataset')
            lnk_abs = pm.Deterministic('lnk_abs', mu_lnk_abs + sigma_lnk_abs*lnk_abs_offset, dims='dataset')

            lnVd_offset = pm.Normal('lnVd_offset', mu=0, sigma=1, dims='dataset')
            lnVd = pm.Deterministic('lnVd', mu_lnVd + sigma_lnVd*lnVd_offset, dims='dataset')

            lnCLC_offset = pm.Normal('lnCLC_offset', mu=0, sigma=1, dims='dataset')
            lnCLC = pm.Deterministic('lnCLC', mu_lnCLC + sigma_lnCLC*lnCLC_offset, dims='dataset')
            #------------


            

            CLC =  pm.math.exp(lnCLC)
            k_abs = pm.Deterministic('k_abs', pm.math.exp(lnk_abs))
            Vd = pm.Deterministic('Vd', pm.math.exp(lnVd))
            k_elim = pm.Deterministic('k_elim', CLC/Vd)

            # Dataset-level PK Params
            pm.Deterministic('Vdss [indiv]', Vd, dims='dataset')
            pm.Deterministic('CLC [indiv]', CLC, dims='dataset')
            pm.Deterministic('halft [indiv]', pm.math.log(2)/k_elim, dims='dataset')
            pm.Deterministic('halft_beta [indiv]', pm.math.log(2)/k_elim, dims='dataset')
            pm.Deterministic('abs_halft [indiv]', pm.math.log(2)/k_abs, dims='dataset')
            if self.fit_fa:
                pm.Deterministic('fa [indiv]', fa, dims='dataset')
            
            
            V = Vd[didx]*BW # Volume for each datapoint



            conc_iv = (D/V)*pm.math.exp(-k_elim[didx]*t)
            conc_abs = fa[didx]*(D/V)*(k_abs[didx]/(k_abs[didx]-k_elim[didx]))*(pm.math.exp(-k_elim[didx]*t) - pm.math.exp(-k_abs[didx]*t))
            conc_ss = pm.Deterministic('conc_ss', Dss/CLC[didx])

            conc = pm.Deterministic('conc', pt.switch(route < 0.5, conc_iv, conc_abs) + conc_ss) # iv == 0, non-iv == 1
            lnconc = pm.Deterministic('lnconc', pm.math.log(conc))
            if self.likelihood == 'Lognormal':
                sigma = pm.Exponential('sigma', 1, dims='study')
                pm.Lognormal('conc_sample', mu=lnconc, sigma=sigma[sidx]) # Create generic 'conc_sample' likelihood for posterior-predicitive checks
            elif self.likelihood == 'Normal':
                cv = pm.Exponential('cv', 1, dims='study')
                pm.Normal('conc_sample', mu=conc, sigma=cv[sidx]*conc) # Create generic 'conc_sample' likelihood for posterior-predicitive checks


            # Parse the individual vs. summary
            indiv_conc = conc[indiv_idx]
            lnindiv_conc = pm.math.log(indiv_conc)
            
            #summary_conc = conc[~indiv_idx]
            #lnsummary_conc = pm.math.log(summary_conc)
            summary_conc = conc[summary_idx]
            lnsummary_conc = pm.math.log(summary_conc)
            Ni_summary = Ni[~indiv_bool]
            

            # Likelihood errors (different for each study)
            if self.likelihood == 'Lognormal':
                
                sigma_indiv = sigma[indiv_sidx]
                sigma_summary = sigma[summary_sidx]
                # Likelihood for individual animal data
                pm.Lognormal('conc_indiv', mu=lnindiv_conc, sigma=sigma_indiv, observed=y_indiv)
                
                # Likelihood for summary-level data
                pm.CustomDist('conc_summary', lnsummary_conc, sigma_summary, Ni_summary, sd_summary, logp=my_logp_lognormal, random=my_random_lognormal, observed=y_summary)
                
            elif self.likelihood == 'Normal':
                
                cv_indiv = cv[indiv_sidx]
                cv_summary = cv[summary_sidx]
                # Likelihood for individual animal data
                pm.Normal('conc_indiv', mu=indiv_conc, sigma=cv_indiv*indiv_conc, observed=y_indiv)
            
                # Likelihood for summary-level data
                pm.CustomDist('conc_summary', summary_conc, cv_summary, Ni_summary, sd_summary, logp=my_logp_normal, random=my_random_normal, observed=y_summary)
        return model



    def _2cmpt_hierarchical(self, time, y_obs, sd_obs, route_idx, dose, dose_ss, BW, study_idx, dataset_idx, coords, indiv_bool, Ni):
        dataset_route_dict = collections.OrderedDict()
        for i in range(len(dataset_idx)):
            dataset_route_dict[dataset_idx[i]] = route_idx.values[i]
        dr_idx = np.array(list(dataset_route_dict.values())) # Match the dataset to the route of exposure
        BW_d = pd.DataFrame({'dataset': dataset_idx, 'BW': BW.values}).groupby('dataset')['BW'].mean().values # Average bodyweight for each dataset
        dose_d = pd.DataFrame({'dataset': dataset_idx, 'dose': dose.values}).groupby('dataset')['dose'].mean().values # Average dose for each dataset

        indiv_idx = np.where(indiv_bool)[0]
        summary_idx = np.where(~indiv_bool)[0]

        with pm.Model(coords=coords) as model:
            # use pm.Data(..., mutable=True) to specify variables that can be changed during posterior-predictive check
            t = pm.Data("t", time, mutable=True)
            y_indiv = y_obs.values[indiv_bool] # Observed data from individually reported data
            y_summary = y_obs.values[~indiv_bool] # Observed mean data from summary data
            sd_summary = sd_obs.values[~indiv_bool] # Observed sd data from summary data

            route = pm.Data("route", route_idx, mutable=True)
            D = pm.Data("D", dose, mutable=True)
            Dss = pm.Data("Dss", dose_ss, mutable=True)
            BW = pm.Data("BW", BW, mutable=True)
            sidx = pm.Data('sidx', study_idx, mutable=True)
            indiv_sidx = sidx[indiv_idx]
            summary_sidx = sidx[summary_idx]
            didx = pm.Data('didx', dataset_idx, mutable=True)

            # Population mean priors
            #-----------------------
            mu_lnCLC = pm.Normal('mu_lnCLC', self.CLC_prior['mu'], self.CLC_prior['sd'])
            mu_lnVdss = pm.Normal('mu_lnVdss', self.Vdss_prior['mu'], self.Vdss_prior['sd'])
            
            mu_lnk_cd = pm.Normal('mu_lnk_cd', -4.61, 1.5, initval=-3)
            mu_lnR = pm.Normal('mu_lnR', 4.61, 1.5, initval=3) # R = V1/V2
            #mu_lnk_cd = pm.Normal('mu_lnk_cd', -4.61, 1, initval=-3)
            #mu_lnR = pm.Normal('mu_lnR', 4.61, 1, initval=3) # R = V1/V2
            
            #mu_lnk_cd = pm.Normal('mu_lnk_cd', 0, 1, initval=-3)
            #mu_lnR = pm.Normal('mu_lnR', 0, 1, initval=3) # R = V1/V2
            
            #mu_lnk_abs = pm.Normal('mu_lnk_abs', np.log(47.5), 0.25, initval=3) # absorption t1/2 < 1 hours
            #mu_lnk_abs = pm.Normal('mu_lnk_abs', 2, 0.25, initval=3)
            #mu_lnk_abs = pm.Normal('mu_lnk_abs', np.log(49.2), 0.25, initval=3) # absorption t1/2 < 2 hours (working)
            mu_lnk_abs = pm.Normal('mu_lnk_abs', np.log(81), 0.25, initval=3) # absorption t1/2 < 1 hours

            # fa-fit
            #================
            if self.fit_fa:
                #mu_fa = pm.Beta('mu_fa', 1,1)
                mu_fa = pm.Beta('mu_fa', 6, 1.5)
                kappaMinusTwo = pm.Gamma('kappaMinusTwo', 0.1, 0.1) # Krushke and Vanpaemel "Bayesian Estimation in Hierarchical Models" (also DBDA)
                kappa = pm.Deterministic('kappa', kappaMinusTwo + 2) 
                fa = pm.Beta("fa", alpha=mu_fa*(kappa-2.0)+1.0, beta=(1.0 - mu_fa)*(kappa - 2.0) + 1, dims='dataset')

                #kappa = pm.Gamma('kappa', 1, 1)
                #print(mu_fa.eval())
                #print(kappa.eval())
                #fa = pm.Beta("fa", alpha=mu_fa*kappa, beta=(1.0 - mu_fa)*kappa, dims='dataset')
                #print(fa.eval())

            else:
                #fa = pm.math.ones_like(didx)
                mu_fa = self.default_fa
                fa = pm.math.ones_like(np.unique(dataset_idx))*self.default_fa
            #==================
            

            # Population PK params
            mu_CLC = pm.math.exp(mu_lnCLC)
            mu_k_abs = pm.Deterministic('mu_k_abs', pm.math.exp(mu_lnk_abs))
            mu_R = pm.Deterministic('mu_R', pm.math.exp(mu_lnR))
            mu_k_cd = pm.Deterministic('mu_k_cd', pm.math.exp(mu_lnk_cd))
            mu_Vdss = pm.Deterministic('mu_Vdss', pm.math.exp(mu_lnVdss))
            
            ################
            mu_k_dc = pm.Deterministic('mu_k_dc', mu_R*mu_k_cd)
            mu_Vd = pm.Deterministic('mu_Vd', mu_R*mu_Vdss/(mu_R+1))
            mu_k_elim = pm.Deterministic('mu_k_elim', mu_CLC/mu_Vd)

            mu_beta = pm.Deterministic('mu_beta', 0.5*(mu_k_cd + mu_k_dc + mu_k_elim - pm.math.sqrt((mu_k_cd + mu_k_dc + mu_k_elim)**2 - 4*mu_k_dc*mu_k_elim)))
            mu_alpha = pm.Deterministic('mu_alpha', mu_k_dc*mu_k_elim/mu_beta)

            mu_A_oral = (mu_k_abs/(mu_Vd*BW))*((mu_k_dc-mu_alpha)/((mu_k_abs-mu_alpha)*(mu_beta-mu_alpha)))
            mu_B_oral = (mu_k_abs/(mu_Vd*BW))*((mu_k_dc-mu_beta)/((mu_k_abs-mu_beta)*(mu_alpha-mu_beta)))
            mu_A_iv = (1./(mu_Vd*BW))*(mu_alpha-mu_k_dc)/(mu_alpha-mu_beta)
            mu_B_iv = (1./(mu_Vd*BW))*(mu_beta-mu_k_dc)/(mu_beta-mu_alpha)

            mu_A = pm.Deterministic('mu_A', pt.switch(route < 0.5, mu_A_iv, mu_A_oral))
            mu_B = pm.Deterministic('mu_B', pt.switch(route < 0.5, mu_B_iv, mu_B_oral))

            mu_conc_iv = (D)*(mu_A*pm.math.exp(-mu_alpha*t) + mu_B*pm.math.exp(-mu_beta*t))
            mu_conc_abs = mu_fa*(D)*(mu_A*pm.math.exp(-mu_alpha*t) + mu_B*pm.math.exp(-mu_beta*t) - (mu_A+mu_B)*pm.math.exp(-mu_k_abs*t))
            
            pm.Deterministic('mu_conc', pt.switch(route < 0.5, mu_conc_iv, mu_conc_abs)) # iv == 0, non-iv == 1
        
            # mu_A_iv = pm.Deterministic('mu_A_iv', (1./(mu_Vd*BW_d.mean()))*(mu_alpha-mu_k_dc)/(mu_alpha-mu_beta))
            # mu_B_iv = pm.Deterministic('mu_B_iv', (1./(mu_Vd*BW_d.mean()))*(mu_beta-mu_k_dc)/(mu_beta-mu_alpha))
            # mu_A_oral = pm.Deterministic('mu_A_oral', ((mu_k_dc-mu_alpha)/((mu_k_abs-mu_alpha)*(mu_beta-mu_alpha)))) # (mu_k_abs/(mu_Vd*BW_d))*((mu_k_dc-mu_alpha)/((mu_k_abs-mu_alpha)*(mu_beta-mu_alpha))))
            # mu_B_oral = pm.Deterministic('mu_B_oral', ((mu_k_dc-mu_beta)/((mu_k_abs-mu_beta)*(mu_alpha-mu_beta)))) # (mu_k_abs/(mu_Vd*BW_d))*((mu_k_dc-mu_beta)/((mu_k_abs-mu_beta)*(mu_alpha-mu_beta))))
            # t_phase = pm.Deterministic('t_phase', pm.math.log(mu_A_iv/mu_B_iv)/(mu_alpha - mu_beta))
            # pm.Deterministic('C_phase', np.mean(dose_d)*(mu_A_iv*pm.math.exp(-mu_alpha*t_phase) + mu_B_iv*pm.math.exp(-mu_beta*t_phase))) # Assume D=mean of applied doses
            ################

            # Population PK parameters
            pm.Deterministic('halft [pop]', pm.math.log(2)*pm.math.exp(mu_lnVdss)/pm.math.exp(mu_lnCLC)) # d
            pm.Deterministic('halft_beta [pop]', pm.math.log(2)/mu_beta) # d
            pm.Deterministic('CLC [pop]', pm.math.exp(mu_lnCLC)) # l/kg/d
            pm.Deterministic('Vdss [pop]', pm.math.exp(mu_lnVdss)) # l/kg
            pm.Deterministic('halft_alpha [pop]', pm.math.log(2)/mu_alpha)
            pm.Deterministic('abs_halft [pop]', pm.math.log(2)/mu_k_abs)
            pm.Deterministic('Vc [pop]', mu_R*mu_Vdss/(mu_R+1)) # Central compartment volume (L/kg)
            if self.fit_fa:
                pm.Deterministic('fa [pop]', mu_fa)

            # Population standard-deviation priors
            #-------------------------------------
            SD = self.pop_SD_prior
            SD2 = 0.2 #0.2 or 0.1
            # sigma_lnCLC = pm.Exponential('sigma_lnCLC', 1./CV)
            # sigma_lnhalft = pm.Exponential('sigma_lnk_cd', 1./CV)
            # sigma_lnVdss = pm.Exponential('sigma_lnVdss', 1./CV)
            # sigma_lnR = pm.Exponential('sigma_lnR', 1./CV)
            # sigma_lnk_abs = pm.Exponential('sigma_lnk_abs', 1./CV)
            sigma_lnCLC = pm.HalfNormal('sigma_lnCLC', SD)
            sigma_lnk_cd = pm.HalfNormal('sigma_lnk_cd', self.Rkcd_SD_prior)
            sigma_lnVdss = pm.HalfNormal('sigma_lnVdss', SD)
            sigma_lnR = pm.HalfNormal('sigma_lnR', self.Rkcd_SD_prior)
            sigma_lnk_abs = pm.HalfNormal('sigma_lnk_abs', self.abs_SD_prior)

            # Population geometric standard deviation parameters where GSD = e^sigma
            # halft = f(CLC, k_cd, Vdss, R)
            pm.Deterministic('CLC [GSD]', pm.math.exp(sigma_lnCLC))
            pm.Deterministic('Vdss [GSD]', pm.math.exp(sigma_lnVdss))
            pm.Deterministic('k_cd [GSD]', pm.math.exp(sigma_lnk_cd))
            pm.Deterministic('R [GSD]', pm.math.exp(sigma_lnR))
            pm.Deterministic('halft [GSD]', pm.math.exp(pm.math.sqrt(sigma_lnCLC**2 + sigma_lnVdss**2 + sigma_lnk_cd**2 + sigma_lnR**2)))

            # Optional log-standard deviation for looking at correlations
            pm.Deterministic('log_sigma_lnCLC', pm.math.log(sigma_lnCLC))
            pm.Deterministic('log_sigma_lnk_cd', pm.math.log(sigma_lnk_cd))
            pm.Deterministic('log_sigma_lnVdss', pm.math.log(sigma_lnVdss))
            pm.Deterministic('log_sigma_lnR', pm.math.log(sigma_lnR))
            pm.Deterministic('log_sigma_lnk_abs', pm.math.log(sigma_lnk_abs))

            # daatset-level sampling (reparameterization to help with sampling)
            #------------------------------------------------------------------
            lnCLC_offset = pm.Normal('lnCLC_offset', mu=0, sigma=1, dims='dataset')
            lnCLC = pm.Deterministic('lnCLC', mu_lnCLC + sigma_lnCLC*lnCLC_offset, dims='dataset')

            lnk_cd_offset = pm.Normal('lnk_cd_offset', mu=0, sigma=1, dims='dataset')
            lnk_cd = pm.Deterministic('lnk_cd', mu_lnk_cd + sigma_lnk_cd*lnk_cd_offset, dims='dataset')

            lnVdss_offset = pm.Normal('lnVdss_offset', mu=0, sigma=1, dims='dataset')
            lnVdss = pm.Deterministic('lnVdss', mu_lnVdss + sigma_lnVdss*lnVdss_offset, dims='dataset')

            lnR_offset = pm.Normal('lnR_offset', mu=0, sigma=1, dims='dataset')
            lnR = pm.Deterministic('lnR', mu_lnR + sigma_lnR*lnR_offset, dims='dataset')
            
            lnk_abs_offset = pm.Normal('lnk_abs_offset', mu=0, sigma=1, dims='dataset')
            lnk_abs = pm.Deterministic('lnk_abs', mu_lnk_abs + sigma_lnk_abs*lnk_abs_offset, dims='dataset')
            #------------

            CLC = pm.Deterministic('CLC', pm.math.exp(lnCLC), dims='dataset')
            Vdss = pm.Deterministic('Vdss', pm.math.exp(lnVdss), dims='dataset')
            R = pm.Deterministic('R', pm.math.exp(lnR), dims='dataset')
            k_cd = pm.Deterministic('k_cd', pm.math.exp(lnk_cd), dims='dataset')
            k_abs = pm.Deterministic('k_abs', pm.math.exp(lnk_abs), dims='dataset')
            
            k_dc = pm.Deterministic('k_dc', R*k_cd, dims='dataset')
            Vd = pm.Deterministic('Vd', R*Vdss/(R+1), dims='dataset')
            V2 = pm.Deterministic('V2', Vd/R, dims='dataset')
            k_elim = pm.Deterministic('k_elim', CLC/Vd, dims='dataset')

            beta = pm.Deterministic('beta', 0.5*(k_cd + k_dc + k_elim - pm.math.sqrt((k_cd + k_dc + k_elim)**2 - 4*k_dc*k_elim)), dims='dataset')
            alpha = pm.Deterministic('alpha', k_dc*k_elim/beta, dims='dataset')
        
            A_iv_dataset = pm.Deterministic('A_iv_dataset',  dose_d*(1./(Vd*BW_d))*(alpha-k_dc)/(alpha-beta), dims='dataset')
            B_iv_dataset = pm.Deterministic('B_iv_dataset',  dose_d*(1./(Vd*BW_d))*(beta-k_dc)/(beta-alpha), dims='dataset')
            A_oral_dataset = pm.Deterministic('A_oral_dataset', fa*dose_d*(k_abs/(Vd*BW_d))*((k_dc-alpha)/((k_abs-alpha)*(beta-alpha))), dims='dataset')
            B_oral_dataset = pm.Deterministic('B_oral_dataset',  fa*dose_d*(k_abs/(Vd*BW_d))*((k_dc-beta)/((k_abs-beta)*(alpha-beta))), dims='dataset')
            
            
            #pm.Deterministic('t_iv', pm.math.log(A_iv_dataset/B_iv_dataset)/(alpha - beta), dims='dataset')

            #t_max = pm.Deterministic('t_max', (1./k_abs)*pm.math.log((B_oral_dataset*beta - A_oral_dataset*alpha)/(A_oral_dataset+B_oral_dataset*k_abs)), dims='dataset')
            #t_iv = pm.Deterministic('t_iv',  -(1./(beta-alpha)) * pm.math.log(A_iv_dataset/B_iv_dataset), dims='dataset')
            #t_dataset = pm.Deterministic('t_dataset', pt.switch(dr_idx < 0.5, t_iv, t_iv+t_max), dims='dataset')
            
            #A_dataset = pm.Deterministic('A_dataset', pt.switch(dr_idx < 0.5, A_iv_dataset*dose_d, dose_d*fa*A_oral_dataset), dims='dataset')
            #B_dataset = pm.Deterministic('B_dataset', pt.switch(dr_idx < 0.5, B_iv_dataset*dose_d, dose_d*fa*B_oral_dataset), dims='dataset')
            #t_dataset = pm.Deterministic('t_dataset', pm.math.log(A_dataset/B_dataset)/(alpha - beta), dims='dataset') # Solve: A*exp(-alpha*t) = B*exp(-beta*t)
            #pm.Deterministic('C_dataset', A_dataset*pm.math.exp(-alpha*t_dataset) + B_dataset*pm.math.exp(-beta*t_dataset))
            V = Vd[didx]*BW

            # Dataset-level PK parameters
            pm.Deterministic('CLC [indiv]', CLC, dims='dataset')
            pm.Deterministic('Vdss [indiv]', Vdss, dims='dataset')
            pm.Deterministic('halft [indiv]', pm.math.log(2)*Vdss/CLC, dims='dataset')
            pm.Deterministic('halft_beta [indiv]', pm.math.log(2)/beta, dims='dataset')
            pm.Deterministic('halft_alpha [indiv]', pm.math.log(2)/alpha, dims='dataset')
            pm.Deterministic('abs_halft [indiv]', pm.math.log(2)/k_abs, dims='dataset')
            if self.fit_fa:
                pm.Deterministic('fa [indiv]', fa, dims='dataset')
            
            A_oral = (k_abs[didx]/V)*((k_dc[didx]-alpha[didx])/((k_abs[didx]-alpha[didx])*(beta[didx]-alpha[didx])))
            B_oral = (k_abs[didx]/V)*((k_dc[didx]-beta[didx])/((k_abs[didx]-beta[didx])*(alpha[didx]-beta[didx])))
            A_iv = (1./V)*(alpha[didx]-k_dc[didx])/(alpha[didx]-beta[didx])
            B_iv = (1./V)*(beta[didx]-k_dc[didx])/(beta[didx]-alpha[didx])

            A = pm.Deterministic('A', pt.switch(route < 0.5, A_iv, A_oral))
            B = pm.Deterministic('B', pt.switch(route < 0.5, B_iv, B_oral))

            conc_iv = (D)*(A*pm.math.exp(-alpha[didx]*t) + B*pm.math.exp(-beta[didx]*t))
            conc_abs = fa[didx]*(D)*(A*pm.math.exp(-alpha[didx]*t) + B*pm.math.exp(-beta[didx]*t) - (A+B)*pm.math.exp(-k_abs[didx]*t))
            conc_ss = pm.Deterministic('conc_ss', Dss/CLC[didx])
            
            conc = pm.Deterministic('conc', pt.switch(route < 0.5, conc_iv, conc_abs) + conc_ss) # iv == 0, non-iv == 1
            lnconc = pm.Deterministic('lnconc', pm.math.log(conc))
            if self.likelihood == 'Lognormal':
                sigma = pm.Exponential('sigma', 1, dims='study') # Prior is average error on data of 1 (log-space)
                pm.Lognormal('conc_sample', mu=lnconc, sigma=sigma[sidx])
            elif self.likelihood == 'Normal':
                cv = pm.Exponential('cv', 1, dims='study') # Prior is average CV on data is 1
                pm.Normal('conc_sample', mu=conc, sigma=cv[sidx]*conc)
    
            # Parse the individual vs. summary
            indiv_conc = conc[indiv_idx]
            lnindiv_conc = pm.math.log(indiv_conc)
            
            #summary_conc = conc[~indiv_idx]
            #lnsummary_conc = pm.math.log(summary_conc)
            summary_conc = conc[summary_idx]
            lnsummary_conc = pm.math.log(summary_conc)
            Ni_summary = Ni[~indiv_bool]
            
            # Likelihood error (different for each study)
            if self.likelihood == 'Lognormal':
                sigma_indiv = sigma[indiv_sidx]
                sigma_summary = sigma[summary_sidx]

                # Likelihood for individual animal data
                pm.Lognormal('conc_indiv', mu=lnindiv_conc, sigma=sigma_indiv, observed=y_indiv)
                # Likelihood for summary-level data
                pm.CustomDist('conc_summary', lnsummary_conc, sigma_summary, Ni_summary, sd_summary, logp=my_logp_lognormal, random=my_random_lognormal, observed=y_summary)
            elif self.likelihood == 'Normal':
                cv_indiv = cv[indiv_sidx]
                cv_summary = cv[summary_sidx]
                # Likelihood for individual data
                pm.Normal('conc_indiv', mu=indiv_conc, sigma=cv_indiv*indiv_conc, observed=y_indiv)
            
                # Likelihood for summary-level data
                pm.CustomDist('conc_summary', y_summary, cv_summary, Ni_summary, sd_summary, logp=my_logp_normal, random=my_random_normal, observed=y_summary)
        return model


    def _1cmpt_single(self, time, y_obs, sd_obs, route_idx, dose, dose_ss, BW, indiv_bool, Ni):
        """1-compartment model with no hierarchy (complete pooling of data). All inputs are vectors of length = N concentrations"""
        indiv_idx = np.where(indiv_bool)[0]
        summary_idx = np.where(~indiv_bool)[0]
        with pm.Model() as model:
            t = pm.Data("t", time, mutable=True)
            y_indiv = y_obs.values[indiv_bool] # Observed data from individually reported data
            y_summary = y_obs.values[~indiv_bool] # Observed mean data from summary data
            sd_summary = sd_obs.values[~indiv_bool] # Observed sd data from summary data



            route = pm.Data("route", route_idx, mutable=True)
            D = pm.Data("D", dose, mutable=True)
            Dss = pm.Data("Dss", dose_ss, mutable=True)
            BW = pm.Data("BW", BW, mutable=True)

            lnCLC = pm.Normal('lnCLC', self.CLC_prior['mu'], self.CLC_prior['sd'])
            lnVd = pm.Normal('lnVd', self.Vdss_prior['mu'], self.Vdss_prior['sd'])
            #lnk_abs = pm.Normal('lnk_abs', 2, 0.25, initval=3)
            #lnk_abs = pm.Normal('lnk_abs', np.log(49.2), 0.25, initval=3) # absorption t1/2 < 1 hours
            lnk_abs = pm.Normal('lnk_abs', np.log(81), 0.25, initval=3) # absorption t1/2 < 1 hours
            if self.fit_fa:
                #fa = pm.Beta('fa', 1, 1)
                fa = pm.Beta('fa', 6, 1.5)
            else:
                fa = self.default_fa

            CLC = pm.math.exp(lnCLC)
            #Vd = pm.math.exp(lnVd)
            Vd = pm.Deterministic('Vd', pm.math.exp(lnVd))
            k_elim = pm.Deterministic('k_elim', CLC/Vd)
            k_abs = pm.Deterministic('k_abs', pm.math.exp(lnk_abs))

            # PK params (using 'pop' nomenclature for consistency with hierarchical models)
            pm.Deterministic('halft [pop]', pm.math.log(2)/k_elim) # d
            pm.Deterministic('Vdss [pop]', Vd) # l/kg
            pm.Deterministic('CLC [pop]', pm.math.exp(lnCLC)) # l/kg/d
            pm.Deterministic('abs_halft [pop]', pm.math.log(2)/k_abs) # d
            if self.fit_fa:
                pm.Deterministic('fa [pop]', fa)
            #V = pm.Deterministic('V', Vd*BW)
            V = Vd*BW

            conc_iv = (D/V)*pm.math.exp(-k_elim*t)
            conc_abs = fa*(D/V)*(k_abs/(k_abs-k_elim))*(pm.math.exp(-k_elim*t) - pm.math.exp(-k_abs*t))
            conc_ss = pm.Deterministic('conc_ss', Dss/CLC)
            conc = pm.Deterministic('conc', pt.switch(route < 0.5, conc_iv, conc_abs) + conc_ss) # iv == 0, non-iv == 1
            lnconc = pm.Deterministic('lnconc', pm.math.log(conc))
            if self.likelihood == 'Lognormal':
                sigma = pm.Exponential('sigma', 1)
                pm.Lognormal('conc_sample', mu=lnconc, sigma=sigma)
            elif self.likelihood == 'Normal':
                cv = pm.Exponential('cv', 1)
                pm.Normal('conc_sample', mu=conc, sigma=cv*conc)
            
            # Parse the individual vs. summary
            indiv_conc = conc[indiv_idx]
            lnindiv_conc = pm.math.log(indiv_conc)
            
            #summary_conc = conc[~indiv_idx]
            #lnsummary_conc = pm.math.log(summary_conc)
            summary_conc = conc[summary_idx]
            lnsummary_conc = pm.math.log(summary_conc)
            Ni_summary = Ni[~indiv_bool]
            
            # Likelihood error (different for each study)
            # Individual animal likelihood
            if self.likelihood == 'Lognormal':
                # Likelihood for individual data
                pm.Lognormal('conc_indiv', mu=lnindiv_conc, sigma=sigma, observed=y_indiv)
                # Likelihood for summary-level data
                pm.CustomDist('conc_summary', lnsummary_conc, sigma, Ni_summary, sd_summary, logp=my_logp_lognormal, random=my_random_lognormal, observed=y_summary)
            elif self.likelihood == 'Normal':
                # Likelihood for individual data
                pm.Normal('conc_indiv', mu=indiv_conc, sigma=cv*indiv_conc, observed=y_indiv)

                # Likelihood for summary-level data. Study sigma calculated in my_logp_normal
                pm.CustomDist('conc_summary', summary_conc, cv, Ni_summary, sd_summary, logp=my_logp_normal, random=my_random_normal, observed=y_summary)
        return model

    def _2cmpt_single(self, time, y_obs, sd_obs, route_idx, dose, dose_ss, BW, indiv_bool, Ni):
        """2-compartment model with no hierarchy (complete pooling of data). All inputs are vectors of length = N concentrations"""
        mean_BW = np.mean(BW)
        indiv_idx = np.where(indiv_bool)[0]
        summary_idx = np.where(~indiv_bool)[0]
        with pm.Model() as model:
            t = pm.Data("t", time, mutable=True)            
            y_indiv = y_obs.values[indiv_bool] # Observed data from individually reported data
            y_summary = y_obs.values[~indiv_bool] # Observed mean data from summary data
            sd_summary = sd_obs.values[~indiv_bool] # Observed sd data from summary data

            route = pm.Data("route", route_idx, mutable=True)
            D = pm.Data("D", dose, mutable=True)
            Dss = pm.Data("Dss", dose_ss, mutable=True)
            BW = pm.Data("BW", BW, mutable=True)

            
            lnCLC = pm.Normal('lnCLC', self.CLC_prior['mu'], self.CLC_prior['sd'])
            lnVdss = pm.Normal('lnVdss', self.Vdss_prior['mu'], self.Vdss_prior['sd'])
            #lnk_cd = pm.Normal('lnk_cd', -4.61, 2, initval=-3)
            #lnR = pm.Normal('lnR', 4.61, 2, initval=3) # R = V1/V2
            lnk_cd = pm.Normal('lnk_cd', -4.61, 1.5, initval=-3)
            lnR = pm.Normal('lnR', 4.61, 1.5, initval=3) # R = V1/V2
            #lnk_abs = pm.Normal('lnk_abs', 2, 0.25, initval=3)
            #lnk_abs = pm.Normal('lnk_abs', np.log(49.2), 0.25, initval=3) # absorption t1/2 < 1 hours
            lnk_abs = pm.Normal('lnk_abs', np.log(81), 0.25, initval=3) # absorption t1/2 < 1 hours
            if self.fit_fa:
                #fa = pm.Beta('fa', 1, 1)
                fa = pm.Beta('fa', 6, 1.5)
            else:
                fa = self.default_fa

            # Population PK params
            CLC = pm.math.exp(lnCLC)
            k_abs = pm.Deterministic('k_abs', pm.math.exp(lnk_abs))
            R = pm.Deterministic('R', pm.math.exp(lnR))
            k_cd = pm.Deterministic('k_cd', pm.math.exp(lnk_cd))
            
            k_dc = pm.Deterministic('k_dc', pm.math.exp(lnR)*pm.math.exp(lnk_cd))
            Vd = pm.math.exp(lnR)*pm.math.exp(lnVdss)/(pm.math.exp(lnR)+1)
            V2 = Vd/pm.math.exp(lnR)
            k_elim = CLC/Vd
            
            beta = pm.Deterministic('beta', 0.5*(k_cd + k_dc + k_elim - pm.math.sqrt((k_cd + k_dc + k_elim)**2 - 4*k_dc*k_elim))) 
            alpha = pm.Deterministic('alpha', k_dc*k_elim/beta)

            # Population PK parameters
            pm.Deterministic('halft [pop]', pm.math.log(2)*pm.math.exp(lnVdss)/pm.math.exp(lnCLC)) # d
            pm.Deterministic('halft_beta [pop]', pm.math.log(2)/beta) # d
            pm.Deterministic('CLC [pop]', pm.math.exp(lnCLC)) # l/kg/d
            pm.Deterministic('Vdss [pop]', pm.math.exp(lnVdss)) # l/kg
            pm.Deterministic('halft_alpha [pop]', pm.math.log(2)/alpha)
            pm.Deterministic('abs_halft [pop]', pm.math.log(2)/k_abs) # d
            pm.Deterministic('Vc [pop]', R*pm.math.exp(lnVdss)/(R+1)) # Central compartment volume (L/kg)
            if self.fit_fa:
                pm.Deterministic('fa [pop]', fa)

            #pm.Deterministic('B_oral', (k_abs/(Vd*mean_BW))*((k_dc-beta)/((k_abs-beta)*(alpha-beta))))
            pm.Deterministic('B_dataset', (1./(Vd*mean_BW))*(beta-k_dc)/(beta-alpha))
            V = Vd*BW
            
            A_oral = (k_abs/V)*((k_dc-alpha)/((k_abs-alpha)*(beta-alpha)))
            B_oral = (k_abs/V)*((k_dc-beta)/((k_abs-beta)*(alpha-beta)))
            A_iv = (1./V)*(alpha-k_dc)/(alpha-beta)
            B_iv = (1./V)*(beta-k_dc)/(beta-alpha)

            A = pm.Deterministic('A', pt.switch(route < 0.5, A_iv, A_oral))
            B = pm.Deterministic('B', pt.switch(route < 0.5, B_iv, B_oral))

            conc_iv = (D)*(A*pm.math.exp(-alpha*t) + B*pm.math.exp(-beta*t))
            conc_abs = fa*(D)*(A*pm.math.exp(-alpha*t) + B*pm.math.exp(-beta*t) - (A+B)*pm.math.exp(-k_abs*t))
            conc_ss = pm.Deterministic('conc_ss', Dss/CLC)
            
            conc = pm.Deterministic('conc', pt.switch(route < 0.5, conc_iv, conc_abs) + conc_ss) # iv == 0, non-iv == 1
            lnconc = pm.Deterministic('lnconc', pm.math.log(conc))
            if self.likelihood == 'Lognormal':
                sigma = pm.Exponential('sigma', 1)
                pm.Lognormal('conc_sample', mu=lnconc, sigma=sigma)#, shape=t.eval().shape)
            elif self.likelihood == 'Normal':
                #cv = pm.Exponential('cv', 1)
                cv = pm.Beta('cv', 1,1)
                #cv = 0.5
                #cv = pm.Gamma("tau", alpha=0.001, beta=0.001)
                pm.Deterministic('sigma', cv*conc)
                pm.Normal('conc_sample', mu=conc, sigma=cv*conc)
                #pm.Normal('conc_sample', mu=conc, tau=cv)

            # Parse the individual vs. summary
            indiv_conc = conc[indiv_idx]
            lnindiv_conc = pm.math.log(indiv_conc)
            
            #summary_conc = conc[~indiv_idx]
            #lnsummary_conc = pm.math.log(summary_conc)
            summary_conc = conc[summary_idx]
            lnsummary_conc = pm.math.log(summary_conc)
            Ni_summary = Ni[~indiv_bool]
            
            # Likelihood error (different for each study)
            # Individual animal likelihood
            if self.likelihood == 'Lognormal':
                # Likelihood for individual data
                pm.Lognormal('conc_indiv', mu=lnindiv_conc, sigma=sigma, observed=y_indiv)
                # Likelihood for summary-level data
                pm.CustomDist('conc_summary', lnsummary_conc, sigma, Ni_summary, sd_summary, logp=my_logp_lognormal, random=my_random_lognormal, observed=y_summary)
            
            elif self.likelihood == 'Normal':
                # Likelihood for individual data
                pm.Normal('conc_indiv', mu=indiv_conc, sigma=cv*indiv_conc, observed=y_indiv)
                #pm.CustomDist('conc_indiv', indiv_conc, cv, 1, 0.1, logp=my_logp_normal, random=my_random_normal, observed=y_indiv)           
            
                # Likelihood for summary-level data
                pm.CustomDist('conc_summary', summary_conc, cv, Ni_summary, sd_summary, logp=my_logp_normal, random=my_random_normal, observed=y_summary)
            
        return model 

    def _build(self, model_type):
        """Private function to build the PyMC model used for sampling

        Parameters
        ----------
        model_type : str
            Build a one-compartment (model_type='1-compartment') or a two-compartment (model_type='2-compartment') model
        """
        time_label = self.time_label
        y_obs_label = self.y_obs_label
        sd_obs_label = self.sd_obs_label
        route_label = self.route_label
        dose_label = self.dose_label
        BW_label = self.BW_label
        indiv_label = self.indiv_label
        N_label = self.N_label
        Dss_label = self.Dss_label

        
        # All the logic to decide what type of model we need.
        # If we have more than 2 datasets and/or more than 1 study, use a hierarchical framework unless self.hierarchy=='single'
        if model_type == '2-compartment':
            if ((len(np.unique(self.dataset_idx)) > 2) & (self.hierarchy == 'default')) | ((len(np.unique(self.study_idx)) > 1) & (self.hierarchy == 'default')) | (self.hierarchy == 'multi'):
                print('hierarchical model, 2-compartment')
                self.model = self._2cmpt_hierarchical(self.data[time_label], self.data[y_obs_label], self.data[sd_obs_label], self.data[route_label], self.data[dose_label], self.data[Dss_label], self.data[BW_label], self.study_idx, self.dataset_idx, self.coords, self.data[indiv_label].values, self.data[N_label])
                self.model_structure = 'hierarchical'
            else:
                print('single level model, 2-compartment')
                self.model = self._2cmpt_single(self.data[time_label], self.data[y_obs_label], self.data[sd_obs_label], self.data[route_label], self.data[dose_label], self.data[Dss_label], self.data[BW_label], self.data[indiv_label].values, self.data[N_label])
                self.model_structure = 'single'
        elif model_type == '1-compartment':
            if ((len(np.unique(self.dataset_idx)) > 2) & (self.hierarchy == 'default')) | ((len(np.unique(self.study_idx)) > 1) & (self.hierarchy == 'default')) | (self.hierarchy == 'multi'):
                print('hierarchical model, 1-compartment')
                self.model = self._1cmpt_hierarchical(self.data[time_label], self.data[y_obs_label], self.data[sd_obs_label], self.data[route_label], self.data[dose_label], self.data[Dss_label], self.data[BW_label], self.study_idx, self.dataset_idx, self.coords, self.data[indiv_label].values, self.data[N_label])
                self.model_structure = 'hierarchical'
            else:
                print('single level model, 1-compartment')
                self.model = self._1cmpt_single(self.data[time_label], self.data[y_obs_label], self.data[sd_obs_label], self.data[route_label], self.data[dose_label], self.data[Dss_label], self.data[BW_label], self.data[indiv_label].values, self.data[N_label])
                self.model_structure = 'single'
        else:
            print('model not specified')
        
        # One extra step if we're using nutpie
        if self.sampler == 'nutpie':
            self.compiled_model = nutpie.compile_pymc_model(self.model)

    def _check_metrics(self):
        """Check the rhat (convergence), effective sample size (ESS), and divergences of the posterior and return a warning if something is wrong with any of them

        Checks adapted from https://github.com/pymc-devs/pymc/blob/main/pymc/stats/convergence.py

        Criteria adapted from https://arxiv.org/abs/1903.08008
        """
        self.pass_all_metrics = True
        valid_name = [rv.name for rv in self.model.free_RVs + self.model.deterministics]
        varnames = []
        for rv in self.model.free_RVs:
            rv_name = rv.name
            if pm_util.is_transformed_name(rv_name):
                rv_name2 = pm_util.get_untransformed_name(rv_name)
                rv_name = rv_name2 if rv_name2 in valid_name else rv_name
            if rv_name in self._trace["posterior"]:
                varnames.append(rv_name)
        ess = az.ess(self._trace, var_names=varnames)
        rhat = az.rhat(self._trace, var_names=varnames)

        # Check rhat
        rhat_max = max(val.max() for val in rhat.values())
        self.rhat_max = rhat_max.item()
        if rhat_max > 1.05:
            msg = (
                "The rhat statistic is larger than 1.05 for some "
                "parameters. This indicates problems during sampling. "
                "See https://arxiv.org/abs/1903.08008 for details"
            )
            warnings.warn(msg)
            self.pass_all_metrics = False
        
        # Check ESS
        eff_min = min(val.min() for val in ess.values())
        eff_per_chain = eff_min / self._trace["posterior"].sizes["chain"]
        self.eff_per_chain = eff_per_chain.item()
        if eff_per_chain < 100:
            msg = (
                "The effective sample size per chain is smaller than 100 for some parameters. "
                " A higher number is needed for reliable rhat and ess computation. "
                "See https://arxiv.org/abs/1903.08008 for details"
            )
            warnings.warn(msg)
            self.pass_all_metrics = False
        
        # Check Divergences
        sampler_stats = self._trace.get("sample_stats", None)
        if sampler_stats is None:
            return
        diverging = sampler_stats.get("diverging", None)
        if diverging is None:
            return

        n_div = int(diverging.sum())
        self.N_divergences = n_div
        if n_div == 0:
            return
        msg = f"There were {n_div} divergences after tuning. Increase `target_accept` or reparameterize."
        warnings.warn(msg)
        self.pass_all_metrics = False

    def run_sample_prior(self, var_names=['conc_indiv', 'conc_summary']):
        with self.model:
            self._trace = pm.sample_prior_predictive(samples=10000, var_names=var_names, random_seed=self.random_seed)
    def run_sample_posterior(self, var_names=['conc_indiv', 'conc_summary']):
        with self.model:
            pm.sample_posterior_predictive(self._trace, extend_inferencedata=True, random_seed=self.random_seed, var_names=var_names)

    def sample(self, cores=4, chains=4, tune=5000, draws=5000, random_seed = 24601, trace_suffix=None, save_trace=True,
               target_accept=0.99, init="jitter+adapt_diag", nuts_sampler='numpyro', model_type='2-compartment', hierarchy='default',
               load_trace=False, likelihood='Lognormal', sample_prior=False, sample_posterior=False, prior_only=False, var_names=['conc_indiv', 'conc_summary']):
        """Sample the model and generate the inference_data xarray object that contains the trace

        Parameters
        ----------
        cores : int, optional
            Number of cores to use for sampling (max 1 core per chain), by default 4
        chains : int, optional
            Number of MCMC chains, by default 4
        tune : int, optional
            Number of tuning samples, by default 5000
        draws : int, optional
            Number of posterior draws, by default 5000
        random_seed : int, optional
            random seed for replicating results, by default 24601
        trace_suffix : _type_, optional
            Optional, user-defined, string to add to the end of the trace path for specific simulations, by default None
        save_trace : bool, optional
            If True, save the trace so it can be loaded later, by default True
        target_accept : float, optional
            Step size is tuned in PyMC to approximate this acceptance rate (PyMC.sample for more info), by default 0.99
        init : str, optional
            Initialize for NUTS algorithm (PyMC.sample for more info), by default "jitter+adapt_diag"
        nuts_sampler : str, optional
            Specify the sampler as either 'numpyro', 'blackjax', 'nutpie', or 'default', by default 'numpyro'
        model_type : str, optional
            Specify one (1-compartment) or two (2-compartment) model for fitting, by default '2-compartment'
        hierarchy : str, optional
            Manually change the hierarcy to 'multi' or 'single' to override logic in self._build, by default 'default'
        load_trace : bool, optional
            Load a previously run trace instead of doing the sampling, by default False
        likelihood : str, optional
            Specific a log-normal ('Lognormal') or normal ('Normal') likelihood for the observed data, by default 'Lognormal'
        sample_prior : bool, optional
            Run the prior predictive check before sampling and add to the inference data object, by default False
        sample_posterior : bool, optional
            Run the posterior predictive check after sampling and add to the inference data object, by default False
        prior_only : bool, optional
            Only run a prior predicitve check, by default False
        var_names : list, optional
            List of concentration variable names for prior and posterior predictive checks, by default ['conc_indiv', 'conc_summary']

        """
        
        
        # Likelihood: 'Lognormal' or 'Normal'
        # hierarchy: 'default' (let the logic decide), 'single' (force single level), 'multi' (force multi-level)

        self.model_type = model_type
        self.likelihood = likelihood
        self.sampler = nuts_sampler
        self.hierarchy = hierarchy
        self.random_seed = random_seed

        if model_type == '1-compartment':
            model_abrev = '1cmpt'
        elif model_type == '2-compartment':
            model_abrev = '2cmpt'
        elif model_type == '1-compartment-ode':
            model_abrev = '1cmpt_ode'
        elif model_type == '2-compartment-ode':
            model_abrev = '2cmpt_ode'

        self.trace_fname = '%s_%s_%s_%s'%(self.chemical, self.sex, self.species, model_abrev)
        if trace_suffix is not None:
            self.trace_fname = '%s_%s' % (self.trace_fname, trace_suffix)
        print(self.trace_fname)
        if self.model is None:
            self._build(model_type)

        # New sampler
        if not load_trace:
            if sample_prior:
                # Only sample prior if load_trace=False
                self.run_sample_prior(var_names=var_names)
                if prior_only:
                    return self
            with self.model:
                if self._trace is not None:
                    self._trace.extend(pm.sample(draws=draws, cores=cores,chains=chains, nuts_sampler=nuts_sampler,
                                        target_accept=target_accept, tune=tune,
                                        init=init, random_seed=random_seed, idata_kwargs = {'log_likelihood': True}))
                else:
                    self._trace = pm.sample(draws=draws, cores=cores,chains=chains, nuts_sampler=nuts_sampler,
                                        target_accept=target_accept, tune=tune,
                                                init=init, random_seed=random_seed, idata_kwargs = {'log_likelihood': True})
            
            if sample_posterior:
                self.run_sample_posterior(var_names=var_names)
        
        
            # Add in additional info on likelihoods and then save final trace
            print('Combining likelihoods...')
            # Create a combined log-likehood that adds the 'conc_indiv' and 'conc_summary' likelihoods for model comparison
            # This 'combined' likelihood simply stacks the calculated likelihood at every observed concentration for both conc_indiv and conc_summary
            # https://nbviewer.org/github/OriolAbril/Exploratory-Analysis-of-Bayesian-Models/blob/multi_obs_ic/content/Section_04/Multiple_likelihoods.ipynb
            if (self._trace.log_likelihood.conc_indiv.shape[-1] > 0) & (self._trace.log_likelihood.conc_summary.shape[-1] > 0):
                # When we have both 'conc_indiv' and 'conc_summary' likelihoods
                # Track the indices to make the combined likelihood has the same order as self.data
                idx = list(self.data[self.data[self.indiv_label]].index)
                idx.extend(list(self.data[~self.data[self.indiv_label]].index))
                orig_order = np.argsort(idx) # argsort will show the location of the original order in the stacked array
                log_lik = self._trace.log_likelihood.copy()
                obs_data = self._trace.observed_data.copy()
                post_pred = self._trace.posterior_predictive.copy()

                # Create 'combined' likelihood which represents likelihood for each observation across all the datasets
                # Length of conc_combined_dim_0 will be N conc_indiv + N conc_summary
                
                # Match the order of the original data
                self._trace.log_likelihood['combined'] = xr.concat([log_lik.conc_indiv.rename({'conc_indiv_dim_0':'conc_combined_dim_0'}), log_lik.conc_summary.rename({'conc_summary_dim_0':'conc_combined_dim_0'})], 'conc_combined_dim_0').isel(conc_combined_dim_0=orig_order).assign_coords(conc_combined_dim_0=range(len(orig_order)))
                self._trace.observed_data['combined'] = xr.concat([obs_data.conc_indiv.rename({'conc_indiv_dim_0':'conc_combined_dim_0'}), obs_data.conc_summary.rename({'conc_summary_dim_0':'conc_combined_dim_0'})], 'conc_combined_dim_0').isel(conc_combined_dim_0=orig_order).assign_coords(conc_combined_dim_0=range(len(orig_order)))
                self._trace.posterior_predictive['combined'] = xr.concat([post_pred.conc_indiv.rename({'conc_indiv_dim_2':'conc_combined_dim_0'}), post_pred.conc_summary.rename({'conc_summary_dim_2':'conc_combined_dim_0'})], 'conc_combined_dim_0').isel(conc_combined_dim_0=orig_order).assign_coords(conc_combined_dim_0=range(len(orig_order)))
                
                # Create 'conc_dataset' likelihood which sums the likelihoods for each observation in a daatset
                dataset_map = dict(zip(self.data.index, self.data.dataset_str))
                dataset_conc = self._trace.log_likelihood['combined'].assign_coords(dataset=('conc_combined_dim_0', [dataset_map[obs.item()] for obs in self._trace.log_likelihood['combined']['conc_combined_dim_0']]))
                self._trace.log_likelihood['conc_dataset'] = dataset_conc.groupby('dataset').sum() # Group all the likelihoods by the dataset they're in and sum



            elif (self._trace.log_likelihood.conc_indiv.shape[-1] > 0) & (self._trace.log_likelihood.conc_summary.shape[-1] == 0):
                # Only 'conc_indiv' likelihood
                self._trace.log_likelihood['combined'] = self._trace.log_likelihood.conc_indiv
                self._trace.observed_data['combined'] = self._trace.observed_data.conc_indiv
                self._trace.posterior_predictive['combined'] = self._trace.posterior_predictive.conc_indiv
            elif (self._trace.log_likelihood.conc_indiv.shape[-1] == 0) & (self._trace.log_likelihood.conc_summary.shape[-1] > 0):
                # Only 'conc_summary' likelihood
                self._trace.log_likelihood['combined'] = self._trace.log_likelihood.conc_summary
                self._trace.observed_data['combined'] = self._trace.observed_data.conc_summary
                self._trace.posterior_predictive['combined'] = self._trace.posterior_predictive.conc_summary
            
            with self.model:
                loo_data = az.loo(self._trace, var_name='combined')
                self._trace.observed_data['pareto_k'] = loo_data.pareto_k
            
            if save_trace:
                self._trace.to_netcdf('../traces/%s.nc'%self.trace_fname)#, engine="netcdf4")
        else:
            self._trace = az.from_netcdf('../traces/%s.nc'%self.trace_fname)
        
        print('Checking metrics...')
        self._check_metrics()
        print(f'Pass all metrics = {self.pass_all_metrics}')
        
        print('Assign pareto...')
        self.data['pareto_k'] = self._trace.observed_data.pareto_k.values

        y_true = self._trace.observed_data.combined.values
        y_pred = self._trace.posterior_predictive.combined.stack(sample=("chain", "draw")).values.T
        y_pred_mean = y_pred.mean(axis=0)
        self.MSLE = skm.mean_squared_log_error(y_true, y_pred_mean)

        self.bayes_r2 = az.r2_score(y_true, y_pred)
        return self
    def calc_c_iv(self, dataset_coord):
        def f_ln(t, A, B, alpha, beta):
            return (np.log(A*np.exp(-alpha*t)) - np.log(B*np.exp(-beta*t)))
        def conc(t, A, B, alpha, beta):
            return A*np.exp(-alpha*t) + B*np.exp(-beta*t)
        params = az.summary(self._trace, var_names=['A_iv_dataset', 'B_iv_dataset', 'alpha', 'beta'], coords={'dataset': dataset_coord})
        A, B, alpha, beta = params.iloc[:, 0].values
        root = sco.fsolve(f_ln, 100, args=(A, B, alpha, beta, ), maxfev=1000)[0]
        return root, conc(root, A, B, alpha, beta)

    def calc_c_gavage(self, dataset_coord):
        def f_ln(t, A, B, alpha, beta, ka):
            return (np.log(A*np.exp(-alpha*t) - (A+B)*np.exp(-ka*t)) - np.log(B*np.exp(-beta*t)))
        def fprime_ln(t, A, B, alpha, beta, ka):
            term1 = alpha*A*np.exp(-ka*t)
            term2 = ka*(A+B)*np.exp(-alpha*t)
            term3 = A*(np.exp(-alpha*t) - np.exp(-ka*t)) + B*np.exp(-alpha*t)
            return ((term1 + term2)/(term3)) + beta
        def conc(t, A, B, alpha, beta, ka):
            return A*np.exp(-alpha*t) + B*np.exp(-beta*t) - (A+B)*np.exp(-ka*t)
        params = az.summary(self._trace, var_names=['A_iv_dataset', 'B_iv_dataset', 'alpha', 'beta', 'k_abs'], coords={'dataset': dataset_coord}, kind='stats', stat_focus='median')
        A, B, alpha, beta, ka = params.iloc[:, 0].values
        root = sco.fsolve(f_ln, 10, fprime=fprime_ln, args=(A, B, alpha, beta, ka, ), maxfev=1000)[0]
        return root, conc(root, A, B, alpha, beta, ka)

    def solve_gavage_t(self, A, B, alpha, beta, ka, t_guess):
        def f_ln(t, A, B, alpha, beta, ka):
            return (np.log(A*np.exp(-alpha*t) - (A+B)*np.exp(-ka*t)) - np.log(B*np.exp(-beta*t)))
        def fprime_ln(t, A, B, alpha, beta, ka):
            term1 = alpha*A*np.exp(-ka*t)
            term2 = ka*(A+B)*np.exp(-alpha*t)
            term3 = A*(np.exp(-alpha*t) - np.exp(-ka*t)) + B*np.exp(-alpha*t)
            return ((term1 + term2)/(term3)) + beta
        root = sco.fsolve(f_ln, t_guess, fprime=fprime_ln, args=(A, B, alpha, beta, ka,), maxfev=1000)
        return xr.DataArray(root)
    def solve_iv_t(self, A, B, alpha, beta, t_guess):
        def f_ln(t, A, B, alpha, beta):
            return (np.log(A*np.exp(-alpha*t)) - np.log(B*np.exp(-beta*t)))
        def fprime_ln(t, A, B, alpha, beta):
            return beta - alpha
        root = sco.fsolve(f_ln, t_guess, args=(A, B, alpha, beta, ), maxfev=1000)
        return xr.DataArray(root)

    #=======================================================================================
    # Methods below are auxiliary methods to analyze the trace after we've finished sampling
    #=======================================================================================

    def calc_fa(self, oral_dataset, iv_dataset, plot=False, last_n=3):
        def fit_ke(t_last, c_last):
            p_beta = np.polyfit(t_last, np.log(c_last), 1)
            return p_beta
        oral_df = self.data[self.data.dataset_str == oral_dataset].copy()
        iv_df = self.data[self.data.dataset_str == iv_dataset].copy()
        t_oral, c_oral = oral_df[self.time_label].values, oral_df[self.y_obs_label].values
        t_iv, c_iv = iv_df[self.time_label].values, iv_df[self.y_obs_label].values
        AUC_oral = np.trapz(c_oral, x=t_oral)
        AUC_iv = np.trapz(c_iv, x=t_iv)

        p_oral = fit_ke(t_oral[-last_n:], c_oral[-last_n:])
        print(p_oral)
        k_oral = -p_oral[0]
        AUC_oral_t_inf = c_oral[-1]/k_oral
        p_iv = fit_ke(t_iv[-last_n:], c_iv[-last_n:])
        k_iv = -p_iv[0]
        AUC_iv_t_inf = c_iv[-1]/k_iv
        if plot:
            fig, ax = plt.subplots(1,2)
            t_oral_pred = np.linspace(t_oral[-last_n], t_oral[-1]*2)
            oral_fit = np.polyval(p_oral, t_oral_pred)
            ax[0].plot(t_oral, c_oral, 'o')
            ax[0].plot(t_oral_pred, np.exp(oral_fit), 'r-')
            ax[0].set_yscale('log')

            t_iv_pred = np.linspace(t_iv[-last_n], t_iv[-1]*2)
            iv_fit = np.polyval(p_iv, t_iv_pred)
            ax[1].plot(t_iv, c_iv, 'o')
            ax[1].plot(t_iv_pred, np.exp(iv_fit), 'r-')
            ax[1].set_yscale('log')


        #return (AUC_oral+AUC_oral_t_inf)/(AUC_iv+AUC_iv_t_inf)
        print(AUC_iv_t_inf, AUC_oral_t_inf)
        print(AUC_oral, AUC_iv)
        return (AUC_oral+AUC_oral_t_inf)/(AUC_iv+AUC_iv_t_inf)


    def generate_samples(self, var_names=['CLC [pop]', 'Vdss [pop]'], num_samples=1000):
        posterior = az.extract(self._trace, var_names=var_names, num_samples=num_samples)
        posterior_df = posterior.to_dataframe()[var_names].reset_index(drop=True)
        return posterior_df

    def plot_pk_levels(self, variable='CLC', reported_values = [], mapper=None):
        """_summary_

        Parameters
        ----------
        reported_fa : list, optional
            List of reported fraction absorbed values for comparison, by default []
        """
        def convert_ml(x):
            return x*1000
        var_names = [variable + x for x in [' [pop]', ' [indiv]']]

        axs = az.plot_forest(self._trace, var_names=var_names, hdi_prob=0.9,combined=True,# coords={"dataset": self.gavage_codes}, 
                            ridgeplot_truncate=True, transform=convert_ml, 
                           ridgeplot_quantiles=[.5])

        axs[0].plot(reported_values, [axs[0].get_yticks()[-1]-0.2]*len(reported_values), 'ko', label='Literature Reported')

        #axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        mu_summary = az.summary(self._trace, var_names=[variable + " [pop]"], kind='stats', hdi_prob=0.9, round_to=10)
        title_str = 'Population: %0.2f (%0.2f - %0.2f)' % (mu_summary.iloc[0, 0]*1000, mu_summary.iloc[0, 2]*1000, mu_summary.iloc[0, 3]*1000)
        axs[0].set_title(title_str, fontsize=16)
        axs[0].set_xlabel("Clearance (ml/kg/d)", fontsize=16)
        if mapper is not None:
            labels = [item.get_text() for item in axs[0].get_yticklabels()]
            new_labels = []
            for l in labels:
                l = l.replace('[', '')
                l = l.replace(']', '')
                new_labels.append(mapper[l])
            axs[0].set_yticklabels(new_labels)


    def plot_fa(self, reported_fa = []):
        """_summary_

        Parameters
        ----------
        reported_fa : list, optional
            List of reported fraction absorbed values for comparison, by default []
        """

        axs = az.plot_forest(self._trace, var_names=["mu_fa","^fa"], filter_vars="regex",hdi_prob=0.9,combined=True, coords={"dataset": self.gavage_codes}, 
                            ridgeplot_truncate=True,
                           ridgeplot_quantiles=[.5])

        axs[0].plot(reported_fa, [axs[0].get_yticks()[-1]-0.2]*len(reported_fa), 'ko', label='Literature Reported')

        #axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        mu_fa_summary = az.summary(self._trace, var_names=["mu_fa"], kind='stats', hdi_prob=0.9)
        title_str = 'Population fa: %0.2f (%0.2f - %0.2f)' % (mu_fa_summary.iloc[0, 0], mu_fa_summary.iloc[0, 2], mu_fa_summary.iloc[0, 3])
        axs[0].set_title(title_str, fontsize=16)

        
    
    
    def plot_CLC_diff(self, mapper=None, hdi_prob=0.95, param="CLC [indiv]", units='L'):

        labeller = azl.MapLabeller(var_name_map={param: ""})
        def calc_ml(x):
            return x*1000
        if units == 'ml':
            transform = calc_ml
        else:
            transform = None
        axs = az.plot_forest(self._trace, var_names=[param], hdi_prob=hdi_prob,combined=True, transform=transform, coords={"dataset": self.iv_oral_codes}, labeller=labeller)
        #axs = az.plot_forest(self._trace, var_names=["lnCLC"], hdi_prob=hdi_prob,combined=True, coords={"dataset": self.iv_oral_codes}, )
        #axs = az.plot_forest(self._trace, var_names=["lnCLC_offset"], hdi_prob=hdi_prob,combined=True, coords={"dataset": self.iv_oral_codes}, )
        
        ax = axs[0]
        for i, line in enumerate(ax.get_lines()):
            if i%2:
                line.set_markerfacecolor('C0')
                line.set_marker('^')
                line.set_markersize('6')
            else:
                line.set_markersize('6')

        y_vals = ax.get_yticks()
        spacing = y_vals[1]
        y_vals = list(zip(*(iter(ax.get_yticks()),) * 2))
        for (y_start, y_stop) in y_vals:
            ax.axhspan(y_start-spacing*0.2, y_stop+spacing*0.2, color="k", alpha=0.1)
        
        #title_str = "%s: %s %s" % (self.chemical, self.sex.lower(), self.species)
        title_str = "%s %s" % (self.sex, self.species)
        ax.set_title(title_str, fontsize=20)
        if units == 'ml':
            ax.set_xlabel("Clearance (ml/kg/d)", fontsize=16)
        elif units == 'L':
            ax.set_xlabel("Clearance (L/kg/d)", fontsize=16)

        if mapper is not None:
            labels = [item.get_text() for item in ax.get_yticklabels()]
            new_labels = []
            for l in labels:
                l = l.replace('[', '')
                l = l.replace(']', '')
                new_labels.append(mapper[l])
            ax.set_yticklabels(new_labels)
        #axs = az.plot_forest(self._trace, var_names=["CLC [indiv]"], hdi_prob=0.95,combined=True, coords={"dataset": self.iv_oral_codes}, )

    def ROPE_compare(self, hero_id, hdi_prob=0.95, rope=[0.8, 1.2], compare_type='dose', 
                     param = 'CLC [indiv]', study_mapper=None, markersize=7):
        """"""
        def extract_dose(string):
            return float(string.split('-')[1].split()[0])

        func_dict = {
            "25%": lambda x: np.percentile(x, 25),
            "75%": lambda x: np.percentile(x, 75)
            }
        
        if compare_type == 'dose':
            codes = self.gavage_codes.copy()
            print(codes)
            codes = [x for x in codes if str(hero_id) in x]
            codes = sorted(codes, key=extract_dose)

        dose_pairs = [[codes[0], codes[i]] for i in range(1, len(codes))]
        fig, ax = plt.subplots()
        store_effect = []
        for i, min_max in enumerate(dose_pairs):
            min_dose, max_dose = min_max
            effect = self._trace.posterior[param].sel(dataset=max_dose) / self._trace.posterior[param].sel(dataset=min_dose)
            probability_within_rope = 100*((effect > rope[0]) & (effect <= rope[1])).mean()
            effect_summary = az.summary(effect, hdi_prob=0.95, kind='stats', stat_funcs=func_dict)
            effect_summary['ROPE_prob'] = np.round(probability_within_rope.values, 3)
            
            if (effect_summary.loc[param, 'hdi_2.5%'] > rope[0]) and (effect_summary.loc[param, 'hdi_97.5%'] < rope[1]):
                effect_summary['ROPE'] = 'Accept Null'
            #elif (effect < rope[0]).all() | (effect > rope[1]).all():
            elif (effect_summary.loc[param, 'hdi_2.5%'] > rope[1]) or (effect_summary.loc[param, 'hdi_97.5%'] < rope[0]):
                effect_summary['ROPE'] = 'Reject Null'
            else:
                effect_summary['ROPE'] = 'No Decision'
            
            hero_id, min_dose_str, _ = min_dose.split('-')
            hero_id, max_dose_str, _ = max_dose.split('-')
            #study_lbl = '%s\n%s %s: %s' % (study_mapper[hero_id], sex.lower(), species, dose)
            mgkg_dose = float(min_dose_str.split(' ')[0])
            if study_mapper is not None:
                if mgkg_dose > 1:
                    study_lbl = '%s\n%s vs. %s mg/kg' % (study_mapper[hero_id], int(float(max_dose_str.split(' ')[0])), int(float(min_dose_str.split(' ')[0])))
                else:
                    study_lbl = '%s\n%s vs. %0.1f mg/kg' % (study_mapper[hero_id], int(float(max_dose_str.split(' ')[0])), float(min_dose_str.split(' ')[0]))
            else:
                study_lbl = '\n'.join(min_max)
            effect_summary = effect_summary.rename(index={param:study_lbl})
            store_effect.append(effect_summary)
        effect_df = pd.concat(store_effect)
        display(effect_df)
        ax.margins(y=0.2)
        # Reject Null plots
        if 'Reject Null' in effect_df.ROPE.values:
            l1 = ax.plot(effect_df.loc[effect_df.ROPE == 'Reject Null', 'mean'], effect_df.loc[effect_df.ROPE == 'Reject Null'].index, 'o', color='black', markersize=markersize, label='C$_L$ difference')
            ax.hlines(effect_df.loc[effect_df.ROPE == 'Reject Null'].index, effect_df.loc[effect_df.ROPE == 'Reject Null', '25%'], effect_df.loc[effect_df.ROPE == 'Reject Null', '75%'], color='black')#, linewidth=3)
            ax.hlines(effect_df.loc[effect_df.ROPE == 'Reject Null'].index, effect_df.loc[effect_df.ROPE == 'Reject Null', 'hdi_2.5%'], effect_df.loc[effect_df.ROPE == 'Reject Null', 'hdi_97.5%'], color='black')
        
        # Accept Null plots
        if 'Accept Null' in effect_df.ROPE.values:
            ax.plot(effect_df.loc[effect_df.ROPE == 'Accept Null', 'mean'], effect_df.loc[effect_df.ROPE == 'Accept Null'].index, 'x', color='black', markersize=markersize)
            ax.hlines(effect_df.loc[effect_df.ROPE == 'Accept Null'].index, effect_df.loc[effect_df.ROPE == 'Accept Null', '25%'], effect_df.loc[effect_df.ROPE == 'Accept Null', '75%'], color='black')#, linewidth=3)
            ax.hlines(effect_df.loc[effect_df.ROPE == 'Accept Null'].index, effect_df.loc[effect_df.ROPE == 'Accept Null', 'hdi_2.5%'], effect_df.loc[effect_df.ROPE == 'Accept Null', 'hdi_97.5%'], color='black')
    
        # No decicion plots
        if 'No Decision' in effect_df.ROPE.values:
            l2 = ax.plot(effect_df.loc[effect_df.ROPE == 'No Decision', 'mean'], effect_df.loc[effect_df.ROPE == 'No Decision'].index, 'o', color='black', fillstyle="none", markersize=markersize, label = 'No difference')
            ax.hlines(effect_df.loc[effect_df.ROPE == 'No Decision'].index, effect_df.loc[effect_df.ROPE == 'No Decision', '25%'], effect_df.loc[effect_df.ROPE == 'No Decision', '75%'], color='black', linestyle='-')#, linewidth=3, )
            ax.hlines(effect_df.loc[effect_df.ROPE == 'No Decision'].index, effect_df.loc[effect_df.ROPE == 'No Decision', 'hdi_2.5%'], effect_df.loc[effect_df.ROPE == 'No Decision', 'hdi_97.5%'], color='black', linestyle='--')

        ylims = ax.get_ylim()

        ax.axvspan(*rope, alpha=0.1, color='blue')
        ax.vlines(rope[0], *ylims, linestyle = '-.', color='blue')
        ax.vlines(rope[1], *ylims, linestyle = '-.', color='blue')
        _ = ax.set_ylim(ylims)
        #ax.set_title(pfas, fontsize=16)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    

    def plot_ROPE(self, n=2, hdi_prob=0.95, rope=[0.8, 1.2], mapper=None):
        """Kruschke, 2018 https://journals.sagepub.com/doi/epub/10.1177/2515245918771304 """

        iv_oral_pairs = [self.iv_oral_codes[i:i + n] for i in range(0, len(self.iv_oral_codes), n)]
        ncols = int(np.ceil(np.sqrt(len(iv_oral_pairs))))
        nrows = int(np.ceil((len(iv_oral_pairs))/float(ncols)))
        if (ncols*nrows > len(iv_oral_pairs)) and (len(iv_oral_pairs) < 4):
            ncols = len(iv_oral_pairs)
            nrows = 1

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
        try:
            all_ax = axs.flatten()
        except:
            all_ax = [axs]
        
        axs = self._trim_axis(all_ax, len(iv_oral_pairs))
        store_rope = {}
        for i, iv_oral in enumerate(iv_oral_pairs):
            iv, oral = iv_oral
            param = 'CLC [indiv]'
            #param = 'lnCLC'
            #diff = (self._trace.posterior[param].sel(dataset=iv) - self._trace.posterior[param].sel(dataset=oral))/ self._trace.posterior[param].sel(dataset=iv)
            diff = self._trace.posterior[param].sel(dataset=iv)/self._trace.posterior[param].sel(dataset=oral)
            
            poolsd = ((self._trace.posterior[param].sel(dataset=iv).std()**2 + self._trace.posterior[param].sel(dataset=oral).std()**2)/2.)**0.5
            #poolsd = np.sqrt( ( self._trace.posterior['CLC [indiv]'].sel(dataset=iv).std()**2 
            #     + self._trace.posterior['CLC [indiv]'].sel(dataset=oral).std()**2 ) / 2 )
            #poolsd = self._trace.posterior['CLC [GSD]']
            #effect = diff#/poolsd
            effect = diff
            probability_within_rope = 100*((effect > rope[0]) & (effect <= rope[1])).mean()

            az.plot_posterior(effect, rope=rope, ax=axs[i], hdi_prob=hdi_prob)
            title_split = iv.split('-')
            title_str = '%s: %s' % (title_split[0], title_split[1])
            if mapper is not None:
                title_str = mapper[title_str]

            store_rope[title_str] = np.round(probability_within_rope.values, 2)
            axs[i].set_title(title_str, fontsize=18)
        fig.tight_layout()
        return pd.Series(store_rope)


    def plot_ref(self, n=2, hdi_prob=0.95):
        iv_oral_pairs = [self.iv_oral_codes[i:i + n] for i in range(0, len(self.iv_oral_codes), n)]
        ncols = int(np.ceil(np.sqrt(len(iv_oral_pairs))))
        nrows = int(np.ceil((len(iv_oral_pairs))/float(ncols)))
        if (ncols*nrows > len(iv_oral_pairs)) and (len(iv_oral_pairs) < 4):
            ncols = len(iv_oral_pairs)
            nrows = 1

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
        try:
            all_ax = axs.flatten()
        except:
            all_ax = [axs]
        
        axs = self._trim_axis(all_ax, len(iv_oral_pairs))
        store_fraction = {}
        for i, iv_oral in enumerate(iv_oral_pairs):
            iv, oral = iv_oral
            print(iv, oral)
            param = 'CLC [indiv]'
            #param = 'lnCLC'
            diff = self._trace.posterior[param].sel(dataset=iv) - self._trace.posterior[param].sel(dataset=oral)
            
            poolsd = ((self._trace.posterior[param].sel(dataset=iv).std()**2 + self._trace.posterior[param].sel(dataset=oral).std()**2)/2.)**0.5
            #poolsd = np.sqrt( ( self._trace.posterior['CLC [indiv]'].sel(dataset=iv).std()**2 
            #     + self._trace.posterior['CLC [indiv]'].sel(dataset=oral).std()**2 ) / 2 )
            #poolsd = self._trace.posterior['CLC [GSD]']
            effect = diff#/poolsd

            #az.plot_posterior(effect, rope=[0.5, 1.5], ax=axs[i], hdi_prob=hdi_prob)
            az.plot_posterior(effect, ref_val=0, ax=axs[i], hdi_prob=hdi_prob)
            title_split = iv.split('-')
            title_str = '%s: %s' % (title_split[0], title_split[1])

            IV_frac = (effect>0).sum()/effect.count()
            store_fraction[title_str] =  IV_frac.values

            axs[i].set_title(title_str, fontsize=18)

        return pd.Series(store_fraction)

    def get_fa_stats(self, hdi_prob=0.9):
        """get the fa statistics from the fitted model

        Parameters
        ----------
        hdi_prob : float, optional
            hdi probability, by default 0.9

        Returns
        -------
        pd.DataFrame
            Summary table of fa mean and credible intervals
        """
        if not self.fit_fa:
            print('fa was not fit')
            return
        if self.model_structure == 'hierarchical':
            return az.summary(self._trace, var_names=["mu_fa","^fa"], filter_vars="regex",kind='stats', coords={"dataset": self.gavage_codes}, hdi_prob=hdi_prob)
        else:
            return az.summary(self._trace, var_names=["^fa"], filter_vars="regex",kind='stats', hdi_prob=hdi_prob)

    def get_pk_stats(self, level='pop', pk_stat=None, hdi_prob=0.9, sample='posterior', vol_units='l'):
        """_summary_

        Parameters
        ----------
        level : str, optional
            Get the population level (level = 'pop') or dataset-level (level = 'indiv') pharmacokinetic parameters, by default 'pop'
        pk_stat : _type_, optional
            Slice summary datafram for specific pharmacokinetic parameter, by default None
        hdi_prob : float, optional
            hdi probability for credible intervals, by default 0.9
        sample : str, optional
            Get pk stats on prior (sample == 'prior') or posterior (sample == 'posterior'), by default 'posterior'
        vol_units : str, optional
            Units for volume (l=liters, ml=milliliters), by defalt 'l'

        Returns
        -------
        pd.DataFrame
            summary dataframe
        """

        if sample == 'posterior':
            pk_stats = az.summary(self._trace, var_names=level, filter_vars="regex",kind='stats', hdi_prob=hdi_prob, round_to=5)
        elif sample == 'prior':
            pk_stats = az.summary(self._trace.prior_predictive, var_names=level, filter_vars="regex",kind='stats', hdi_prob=hdi_prob, stat_focus='median', round_to=5)

        if vol_units == 'ml':
            CLC_idx = [x for x in pk_stats.index if 'CLC' in x]
            Vd_idx = [x for x in pk_stats.index if 'Vd' in x]
            pk_stats.loc[CLC_idx] = pk_stats.loc[CLC_idx]*1000
            pk_stats.loc[Vd_idx] = pk_stats.loc[Vd_idx]*1000

        if pk_stat is not None:
            return pk_stats[pk_stats.index.str.startswith(pk_stat)]
        else:
            return pk_stats

    def plot_residuals(self, x='Model', y='Residual', margin=0.1, log_residuals=True, save=True):
        fig, ax = plt.subplots(figsize=(4.25, 3.19), layout='constrained')
        pred = az.extract(self._trace.posterior_predictive, var_names='combined', num_samples=1000)
        tmp_data = self.data.copy()
        

    
        y_data = tmp_data[['hero_id', self.time_label, self.y_obs_label]].copy()
        model_mean = np.mean(pred.values, axis=1)
                
        if y == 'Residual':
            if log_residuals:
                r = np.log(y_data[self.y_obs_label].values[:, np.newaxis]) - np.log(pred.values)
            else:
                r = y_data[self.y_obs_label].values[:, np.newaxis] - pred.values
        elif y == 'Residual/Model':
            if log_residuals:
                r = (np.log(y_data[self.y_obs_label].values[:, np.newaxis]) - np.log(pred.values))/pred.values
            else:
                r = (y_data[self.y_obs_label].values[:, np.newaxis] - pred.values)/pred.values
        r_hdi = az.hdi(r.T)
        r_mean = np.mean(r, axis=1)
        y_data['r_low'] = r_hdi[:,0]
        y_data['r_mean'] = r_mean
        y_data['r_high'] = r_hdi[:,1]
        y_data['model_mean'] = model_mean

        study_labels = tmp_data.hero_id.unique()
        rgb_values = sns.color_palette("viridis", len(study_labels))
        color_map = dict(zip(study_labels, rgb_values))

        #y_data = y_data[(y_data.r_mean < 1) & (y_data.r_mean > -1)]
        if x == 'Time':
            x_plot = y_data[self.time_label]
        elif x == 'Model':
            x_plot = y_data['model_mean']


        xlims = [min(x_plot)*(1-margin), max(x_plot)*(1+margin)]
        ax.plot(xlims, np.repeat(0, len(xlims)), color='black', linestyle='--')

        ax.scatter(x_plot, y_data['r_mean'], s=6, c=y_data['hero_id'].map(color_map))

        xresiduals = np.linspace(min(x_plot), max(x_plot), 300)

        #ax.vlines(x_plot, y_data['r_low'], y_data['r_high'], color=y_data['hero_id'].map(color_map))
        
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_xscale('log')
        ax.set_xlim(xlims)
        
        if len(study_labels) <= 3:
            ncols = len(study_labels)
        else:
            ncols = 3

        legend_patches = [Patch(color=color, label=study) for study,color in color_map.items()]
        fig.legend(handles=legend_patches, loc='outside lower center', ncols=ncols)
        #fig.tight_layout()
        if save:
            fig_name = '%s_%s_%s_res.png' % (self.chemical,self.species, self.sex)
            fig.savefig('../ppc/%s/%s'%('residuals', fig_name), bbox_inches='tight')
        

    def plot_data_ppc(self, group='prior', var_names = ['conc_indiv', 'conc_summary'], display_metrics=True, savefig=False, color_dataset=False):
        """plot the prior or posterior predicitve check against the observed concentrations

        Parameters
        ----------
        group : str, optional
            Run posterior predictive check (group == 'posterior') or prior predictive check (group == 'prior'), by default 'prior'
        var_names : list, optional
            Likelihoods to check, by default ['conc_indiv', 'conc_summary']
        """

        if group == 'prior':
            pred = self._trace.prior_predictive
            stat_focus = 'median'
            hdi_prob = 0.99
        else:
            pred = self._trace.posterior_predictive
            stat_focus = 'mean'
            hdi_prob = 0.95
        
        fig, ax = plt.subplots(len(var_names), 1, figsize=(4.25, 3.19))
        for i, var_name in enumerate(var_names):
            if var_name == 'conc_indiv':
                data = self.data.loc[self.data[self.indiv_label], :]
            else:
                data = self.data.loc[~self.data[self.indiv_label], :]
            print(var_name)
            if data.empty:
                continue
            
            var_summary = az.summary(pred, var_names=var_name, round_to=20, hdi_prob=hdi_prob, kind="stats", stat_focus=stat_focus)
            
            combine = pd.concat([var_summary.reset_index(drop=True), data.reset_index(drop=True)], axis=1).sort_values(by=self.y_obs_label).reset_index(drop=True)
            # Unity plot
            #ax[i].errorbar(combine.iloc[:, 0], combine[self.y_obs_label], xerr=combine[self.sd_obs_label], fmt='o', color='blue')
            #ax[i].vlines(combine.iloc[:, 0], combine.iloc[:, 2], combine.iloc[:, 3], 'k', alpha=0.2, linewidth=3)

            # # Across all data points
            

            # lnsd2 = np.log(1+(df['conc_sd_cor']**2)/(df['conc_mean_cor']**2))
            # df['lnconc_sd'] = np.sqrt(lnsd2)
            # df['lnconc_mean'] = np.log(df['conc_mean_cor']) - 0.5*lnsd2
            
            # Some digitized error results in negative 
            # ax[i].plot(combine.index, combine[self.y_obs_label], 'o', label='Reported')
            # low_error = combine[self.y_obs_label] - combine[self.sd_obs_label]
            # high_error = combine[self.y_obs_label] + combine[self.sd_obs_label]
            # low_error[low_error < 0] = combine[self.y_obs_label]
            # ax[i].vlines(combine.index, low_error, high_error)
            
            if color_dataset:
                sns.scatterplot(x=combine.index, y=combine[self.y_obs_label], hue=combine['dataset_str'], ax=ax[i], palette='tab10')
            else:
                ax[i].plot(combine.index, combine.iloc[:, 0], 'kx', alpha=0.5, label='Predicted')
                ax[i].vlines(combine.index, combine.iloc[:, 2], combine.iloc[:, 3], 'k', alpha=0.2, linewidth=2)
                ax[i].errorbar(combine.index, combine[self.y_obs_label], yerr=combine[self.sd_obs_label], fmt='o', label='Reported', markersize=3)
            
            
            #ax[i].fill_between(combine.index, combine.iloc[:, 2], combine.iloc[:, 3], color='k', alpha=0.2)
            if color_dataset:
                ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            else:
                ax[i].legend(loc='upper left', fontsize=8)
            
            ax[i].set_title(var_name)
            ax[i].set_ylabel('Conc. [ug/ml]', fontsize=12)
            ax[i].set_yscale('log')
        
        if display_metrics and group != 'prior':
            ax[-1].text(0.98, 0.02, "MSLE = %0.3f\nR$^2$ = %0.2f+/-%0.2f"%(self.MSLE, self.bayes_r2['r2'], self.bayes_r2['r2_std']), transform=ax[-1].transAxes, fontsize=12, ha='right', va='bottom')
        
        fig.tight_layout()
        if savefig:
            fig_name = '%s_%s_%s_ppc.png' % (self.chemical,self.species, self.sex)
            fig.savefig('../ppc/%s/%s'%(group, fig_name))

    
    def _calc_terminal(self, t, C, ax, metric = 'half-life'):
        p_beta = np.polyfit(t[-70:], np.log(C[-70:]), 1)
        #p_beta = np.polyfit(t[50:100], np.log(C[50:100]), 1)
        beta_fit = np.polyval(p_beta, t)
        beta_intercept = np.exp(p_beta[1])
        beta_halft = (-np.log(2)/p_beta[0])
        ax.plot(t, np.exp(beta_fit), 'r-')
        if metric == 'half-life':
            anno_str = "terminal halft:\n%0.3f days"%beta_halft
        elif metric == 'intercept':
            anno_str = 'intercept:\n%0.3f mg/L'% beta_intercept
        ax.annotate(anno_str, xy=(0.6, 0.8), xycoords=ax.transAxes)
        return ax
    
    def _plot_intercept(self, t, B, ke, ax):
        t_plot = np.linspace(t.min(), t.max())
        C_plot = -ke*t_plot + B
        ax.plot(t_plot, C_plot, 'r-')
        anno_str = 'intercept:\n%0.3f mg/L'% B
        ax.annotate(anno_str, xy=(0.6, 0.8), xycoords=ax.transAxes)
        return ax

    def _trim_axis(self, axs, N):
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    def plot_summary_ppc(self, hdi_prob=0.9, fold_error=3, plot_pareto=False):
        """Create a plot of identity line for observed vs. predicted concentrations

        Parameters
        ----------
        hdi_prob : float, optional
            hdi probability for credible intervals, by default 0.9
        fold_error : int, optional
            _description_, by default 3
        """

        conc_summary = az.summary(self._trace.posterior,  var_names="conc_sample", hdi_prob=hdi_prob, kind="stats", round_to=20)
        conc_reported = self.data[[self.y_obs_label, self.sd_obs_label, 'pareto_k']]

        fig, ax = plt.subplots(1,1)
        if plot_pareto:
            p_coff = 1
            p_under = conc_reported.index[conc_reported.pareto_k <= p_coff].tolist()
            p_over = conc_reported.index[conc_reported.pareto_k > p_coff].tolist()
            self.conc_summary = conc_summary
            self.conc_reported = conc_reported
            ax.errorbar(conc_reported.loc[p_under, self.y_obs_label], conc_summary.iloc[p_under ,0], xerr=conc_reported.loc[p_under, self.sd_obs_label], fmt='o', color='black', markersize=5)
            ax.errorbar(conc_reported.loc[p_over, self.y_obs_label], conc_summary.iloc[p_over,0], xerr=conc_reported.loc[p_over, self.sd_obs_label], fmt='^', color='red', markersize=5)

            ax.vlines(conc_reported.loc[p_under, self.y_obs_label], conc_summary.iloc[p_under,2], conc_summary.iloc[p_under,3], color='black')
            ax.vlines(conc_reported.loc[p_over, self.y_obs_label], conc_summary.iloc[p_over,2], conc_summary.iloc[p_over,3], color='red')
        else:
            ax.errorbar(conc_reported[self.y_obs_label], conc_summary.iloc[:,0], xerr=conc_reported[self.sd_obs_label], fmt='o', color='black', markersize=5)
            ax.vlines(conc_reported[self.y_obs_label], conc_summary.iloc[:,2], conc_summary.iloc[:,3], color='black')
        ax.set_yscale('log')
        ax.set_xscale('log')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        MSLE = np.round(skm.mean_squared_log_error(conc_reported[self.y_obs_label], conc_summary.iloc[:,0]), 3)
        R2 = np.round(skm.r2_score(conc_reported[self.y_obs_label], conc_summary.iloc[:,0]), 3)
        axis_min = max(min(xlim[0], ylim[0]), min((conc_reported[self.y_obs_label]-conc_reported[self.sd_obs_label]).min(), conc_summary.iloc[:,0].min()))
        axis_max = max(xlim[1], ylim[1])
        #axis_min = 10**(np.floor(np.log10(axis_min)))
        #axis_max = 10**(np.ceil(np.log10(axis_max)))

        ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k', linewidth=2)
        ax.plot([axis_min, axis_max], [axis_min*(fold_error**1), axis_max*(fold_error**1)], 'k--', linewidth=2)
        ax.plot([axis_min, axis_max], [axis_min*(fold_error**-1), axis_max*(fold_error**-1)], 'k--', linewidth=2)
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)


        ax.set_xlabel('Meas. Conc. [mg/L]', fontsize=16)
        ax.set_ylabel('Pred. Conc. [mg/L]', fontsize=16)

        
        ax.text(0.05, 0.85, "$MSLE = %0.3f$\n$R^2 = %0.3f$"%(self.MSLE, self.bayes_r2), transform=ax.transAxes, fontsize=14)


    def coplot(self, hero_id = None, tf=None, thin=5, var_names=['conc_sample'], hdi_prob=0.95):
        if hero_id:
            full_data = self.data[self.data.hero_id == hero_id].copy()
        else:
            full_data = self.data.copy()
        dataset_codes = full_data.dataset_str.unique()
        
        fig, (ax, cbar_ax) = plt.subplots(ncols=2, figsize=(4, 4), gridspec_kw={'width_ratios': [5, 1]})
        cmap = plt.cm.viridis
        #cmap = plt.cm.seismic
        dose_min, dose_max = full_data[self.dose_label].min(), full_data[self.dose_label].max()
        #norm = plt.LogNorm(vmin=dose_min, vmax=dose_max)
        norm = mpl.colors.LogNorm(vmin=dose_min, vmax=dose_max)
        print(dose_min, dose_max)
        #norm = mpl.colors.PowerNorm(0.1, vmin=dose_min, vmax=dose_max)
        if not tf:
            tf = self.data[self.time_label].max()#*3

        dose_plot = []
        conc_plot = []
        for i, dataset in enumerate(dataset_codes):
            #tmp_df = self.data[self.data.dataset_str == dataset].copy()
            tmp_df = full_data[full_data.dataset_str == dataset].copy()
            params = self._build_ppc_data(tmp_df, tf=tf)
            test_t = params['t']
            if thin is not None:
                thinned_idata = self._trace.sel(draw=slice(None,None, thin))
            else:
                thinned_idata = self._trace
            print('Running PPC...')
            with self.model:
                print('setting params...')
                pm.set_data(params)
                print('sampling...')
                dataset_ppc = pm.sample_posterior_predictive(thinned_idata, var_names = var_names)
            self.dataset_ppc = dataset_ppc
            print('Calculating stats...')
            
            # Include the IQR in the stats summary
            func_dict = {
                "25%": lambda x: np.percentile(x, 25),
                "75%": lambda x: np.percentile(x, 75)
                }
            dose = float(dataset.split('-')[1].split(' ')[0])
            dose_plot.append(dose)
            
            conc_summary = az.summary(dataset_ppc.posterior_predictive,  var_names=var_names, hdi_prob=hdi_prob, kind="stats", round_to=20, stat_funcs=func_dict)
            conc_plot.append(conc_summary.iloc[-1, 0])
            ax.plot(test_t, conc_summary.iloc[:, 0], linewidth=3, color=cmap(norm(dose)), label=dataset)
            #if i>1:
            #    break
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cbar_ax.set_ylabel('Dose (mg/kg)', rotation=270)
        ax.set_xlabel('time [days]', fontsize=18)
        ax.set_ylabel('Pred. Conc. [mg/L]', fontsize=18)
        ax.set_yscale('log')
        #ax.legend(loc='best')
        fig.tight_layout()

        fig1, ax1 = plt.subplots()
        self.dose_plot = dose_plot
        self.conc_plot = conc_plot
        ax1.loglog(dose_plot, conc_plot, 'o')
        ax1.set_xlabel('dose (mg/kg)')
        ax1.set_ylabel('C_final (mg/L)')
    
    def _repeat_dosing(self, dose,  route, n_doses, n_times=100, tau=1, BW=0.25, thin=5):
        # NOT WORKING
        test_t = np.linspace(0, tau, n_times)
        all_t = np.linspace(0, n_doses*tau, n_doses*n_times)
        test_route = np.array([route]*len(test_t))
        test_dose = np.array([dose]*len(test_t))
        test_BW = np.array([BW]*len(test_t))
        if thin is not None:
            thinned_idata = self._trace.sel(draw=slice(None,None,thin))
        else:
            thinned_idata = self._trace
        params = {'t': test_t,
                    #'y': test_y,
                    'route': test_route,
                    'D': test_dose,
                    'BW': test_BW,
                    }
        with self.model:
            pm.set_data(params)
            ppc = pm.sample_posterior_predictive(thinned_idata, var_names = ['mu_conc'])
        func_dict = {
                    "25%": lambda x: np.percentile(x, 25),
                    "75%": lambda x: np.percentile(x, 75)
                    }
        
        conc_summary = az.summary(ppc.posterior_predictive,  var_names=['mu_conc'], hdi_prob=0.9, kind="stats", round_to=20, stat_funcs=func_dict)
        self.conc_summary = conc_summary
        cummulative_list = [conc_summary]
        for i in range(1,n_doses):
            #cummulative_list.append(conc_summary*(i+1))
            cummulative_list.append(conc_summary + cummulative_list[i-1].iloc[-1,:])
        repeat_conc = pd.concat(cummulative_list)
        print(np.shape(all_t))
        print(repeat_conc.shape)
        self.repeat_conc = repeat_conc
        repeat_conc['time'] = all_t
        
        low, med, high = repeat_conc.iloc[:, 2], repeat_conc.iloc[:, 0], repeat_conc.iloc[:, 3]
        low_quant, high_quant = repeat_conc.iloc[:, 4], repeat_conc.iloc[:, 5]
        fig, ax = plt.subplots()
        ax.plot(all_t, med, 'k', linewidth=3)
        ax.fill_between(all_t, low_quant, high_quant, color='blue', alpha=0.5)
        ax.fill_between(all_t, low, high, color='blue', alpha=0.1)
    def _study_ppc(self, hero_id, pop_stat='CLC', study_stat='CLC', thin=5, continuous=True, fold_error=3, plot_pareto=False, plot_intercept=False, custom_grid = None,
                   ts=0, tf=None, plot_terminal=None, plot_transition=False, flat_grid=False, constant_ax = False, var_names=['conc_sample'], hdi_prob=0.9, group='posterior'):
        """Run a time course posterior-predictive check after MCMC for all datasets in the given hero_id

        Parameters
        ----------
        hero_id : int
            hero_id containing the datasets
        pop_stat : str, optional
            _description_, by default 'CLC'
        study_stat : str, optional
            _description_, by default 'CLC'
        thin : int, optional
            Number to thin posterior distributon for increased speed in posterior predictive sample, by default 5
        continuous : bool, optional
            If continuous=True, plot the time-course posterior prediction for all datasets in hero_id. If continuous=False plot an identify line for all datasets in hero_id, by default True
        fold_error : float, optional
            If making a unity plot, plot +/- fold error with dashed lines
        ts : int, optional
            _description_, by default 0
        tf : _type_, optional
            _description_, by default None
        plot_terminal : _type_, optional
            'half-life' or 'intercept', by default None
        flat_grid : bool, optional
            _description_, by default False
        constant_ax : bool, optional
            Maintain same y-axis across all datasets, by default False
        var_names : list, optional
            Random variable from model to plot, by default ['conc_sample']
        hdi_prob : float, optional
            _description_, by default 0.95

        Returns
        -------
        _type_
            _description_
        """

        hero_df = self.data[self.data.hero_id == hero_id].copy()
        hero_df = hero_df.sort_values(by=[self.dose_label, self.route_label], ascending=False)
        datasets = hero_df.dataset_str.unique()
        if flat_grid:
            ncols = len(datasets)
            nrows = 1
        elif custom_grid is not None:
            nrows, ncols = custom_grid
        else:
            ncols = int(np.ceil(np.sqrt(len(datasets))))
            nrows = int(np.ceil((len(datasets))/float(ncols)))
            if (ncols*nrows > len(datasets)) and (len(datasets) < 4):
                ncols = len(datasets)
                nrows = 1

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4), sharey=constant_ax, sharex=constant_ax)
        try:
            all_ax = axs.flatten()
        except:
            all_ax = [axs]
        
        axs = self._trim_axis(all_ax, len(datasets))
        ax_min = 0.5
        ax_max = 1.5
        pop_vars = ['halft [pop]', 'Vdss [pop]', 'CLC [pop]']
        indiv_vars = ['halft [indiv]', 'Vdss [indiv]', 'CLC [indiv]']
        if self.model_type == '2-compartment':
            pop_vars.append('halft_beta [pop]')
            indiv_vars.append('halft_beta [indiv]')
        
        pop_stats = az.summary(self._trace, var_names = pop_vars, hdi_prob=hdi_prob, round_to=6)
        if self.model_structure == 'hierarchical':
            indiv_stats = az.summary(self._trace, var_names = indiv_vars, hdi_prob=hdi_prob, round_to=6)
        
        for i, dataset in enumerate(datasets):
            ax = all_ax[i]
            tmp_df = hero_df[hero_df.dataset_str == dataset]
            params = self._build_ppc_data(tmp_df)
            test_t = params['t']
            self.params = params
            if continuous:        
                if plot_pareto:
                    p_coff = 1 # Pareto k cutoff
                    ax.errorbar(tmp_df.loc[tmp_df.pareto_k > p_coff, self.time_label], tmp_df.loc[tmp_df.pareto_k > p_coff, self.y_obs_label], yerr=tmp_df.loc[tmp_df.pareto_k > p_coff, self.sd_obs_label], fmt='r^')
                    ax.errorbar(tmp_df.loc[tmp_df.pareto_k <= p_coff, self.time_label], tmp_df.loc[tmp_df.pareto_k <= p_coff, self.y_obs_label], yerr=tmp_df.loc[tmp_df.pareto_k <= p_coff, self.sd_obs_label], fmt='ko')
                else:
                    ax.errorbar(tmp_df[self.time_label], tmp_df[self.y_obs_label], yerr=tmp_df[self.sd_obs_label], fmt='ko')
                
                if thin is not None:
                    thinned_idata = self._trace.sel(draw=slice(None,None,thin))
                else:
                    thinned_idata = self._trace
                print('Running PPC...')
                with self.model:
                    print('setting params...')
                    pm.set_data(params)
                    print('sampling...')
                    if group == 'posterior':
                        dataset_ppc = pm.sample_posterior_predictive(thinned_idata, var_names = var_names)
                    elif group == 'prior':
                        dataset_ppc = pm.sample_prior_predictive(samples=2000, var_names = var_names, random_seed=self.random_seed)
                self.dataset_ppc = dataset_ppc
                print('Calculating stats...')
                
                # Include the IQR in the stats summary
                func_dict = {
                    "25%": lambda x: np.percentile(x, 25),
                    "75%": lambda x: np.percentile(x, 75)
                    }
                if group == 'posterior':
                    conc_summary = az.summary(dataset_ppc.posterior_predictive,  var_names=var_names, hdi_prob=hdi_prob, kind="stats", round_to=20, stat_funcs=func_dict)
                elif group == 'prior':
                    conc_summary = az.summary(dataset_ppc.prior,  var_names=var_names, hdi_prob=0.95, kind="stats", stat_focus='median', round_to=20, stat_funcs=func_dict)
                
                self.conc_summary = conc_summary
                low, med, high = conc_summary.iloc[:, 2], conc_summary.iloc[:, 0], conc_summary.iloc[:, 3]
                low_quant, high_quant = conc_summary.iloc[:, 4], conc_summary.iloc[:, 5]
                
                ax.plot(test_t, med, 'k', linewidth=3)
                #ax.plot(test_t, low, 'b--', linewidth=2)
                #ax.plot(test_t, high, 'b--', linewidth=2)
                if group == 'posterior':
                    ax.fill_between(test_t, low_quant, high_quant, color='blue', alpha=0.5)
                ax.fill_between(test_t, low, high, color='blue', alpha=0.1)

                if plot_terminal:
                    ax = self._calc_terminal(test_t, med, ax, metric=plot_terminal)
                if plot_intercept:
                    int_summary = az.summary(self._trace, var_names = ['B_dataset', 'halft [indiv]'], coords={'dataset': dataset})
                    B = int_summary.loc['B_dataset', 'mean']
                    ke = np.log(2)/int_summary.loc['halft [indiv]', 'mean']
                    ax = self._plot_intercept(test_t, B, ke, ax)
                if plot_transition:
                    if 'iv' in dataset:
                        t, c = self.calc_c_iv(dataset)
                    else:
                        t, c = self.calc_c_iv(dataset)
                    ax.plot(t, c, 'go')
                    ax.plot([test_t[0], t], [c, c], 'g')
                    ax.plot([t, t], [ax.get_ylim()[0], c], 'g')
                ax.set_xlabel('time [days]', fontsize=18)
                ax.set_ylabel('Meas. Conc. [mg/L]', fontsize=18)
                ax.set_yscale('log')
            else:
                conc_post = az.summary(self._trace.posterior, var_names='conc_sample', hdi_prob=hdi_prob, kind="stats").reset_index(drop=True)
                conc_summary = conc_post.loc[tmp_df.index]

                #ax.plot(tmp_df['conc_mean'], conc_summary['mean'], 'ko')
                #ax.vlines(tmp_df['conc_mean'], conc_summary.iloc[:, 2], conc_summary.iloc[:, 3], 'k')
                ax.plot(tmp_df[self.y_obs_label], conc_summary.iloc[:, 0], 'ko')
                ax.vlines(tmp_df[self.y_obs_label], conc_summary.iloc[:, 2], conc_summary.iloc[:, 3], 'k')
                xlim, ylim = ax.get_xlim(), ax.get_ylim()

                axis_min = max(min(xlim[0], ylim[0]), min((tmp_df[self.y_obs_label]-tmp_df[self.sd_obs_label]).min(), conc_summary.iloc[:,0].min()))
                axis_max = max(xlim[1], ylim[1])

                ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k', linewidth=2)
                ax.plot([axis_min, axis_max], [axis_min*(fold_error**1), axis_max*(fold_error**1)], 'k--', linewidth=2)
                ax.plot([axis_min, axis_max], [axis_min*(fold_error**-1), axis_max*(fold_error**-1)], 'k--', linewidth=2)

                ax.set_xlim(axis_min, axis_max)
                ax.set_ylim(axis_min, axis_max)
                
                
                ax.set_xlabel('Meas. Conc. [mg/L]', fontsize=18)
                ax.set_ylabel('Pred. Conc. [mg/L]', fontsize=18)
                ax.set_xscale('log')
                ax.set_yscale('log')
            if self.model_structure == 'hierarchical':
                stat = indiv_stats.loc['%s [indiv][%s]'%(study_stat,dataset)]
                low_stat, med_stat, high_stat = stat[2], stat[0], stat[3]
                
                #study_title_str = 'Hero: %s\n%s (%s): %.3g (%.3g - %.3g)' % (dataset, study_stat, self.units[study_stat], med_stat*1000, low_stat*1000, high_stat*1000)
                if group == 'posterior':
                    study_title_str = 'Hero: %s\n%s (%s): %.3g (%.3g - %.3g)' % (dataset, study_stat, self.units[study_stat], med_stat*1000, low_stat*1000, high_stat*1000)
                elif group == 'prior':
                    study_title_str = 'Hero: %s' % dataset
                #study_title_str = '%s\n%.5g (%.5g - %.5g)' % ('-'.join(dataset.split('-')[1:]),  med_stat, low_stat, high_stat)
                ax.set_title(study_title_str, fontsize=14)
            else:
                study_title_str = 'Hero: %s'
                ax.set_title(study_title_str%(dataset))
            ax_lim = ax.get_ylim()
            if ax_min > ax_lim[0]:
                ax_min = ax_lim[0]
            if ax_max < ax_lim[1]:
                ax_max = ax_lim[1]
            #break
        mu_stat = pop_stats.loc['%s [pop]'%pop_stat]
        low_stat, med_stat, high_stat = mu_stat[2], mu_stat[0], mu_stat[3]
        if group == 'posterior':
            pop_title_str = "Population %s (%s): %.3g (%.3g - %.3g)" % (self.description[pop_stat], self.units[pop_stat], med_stat*1000, low_stat*1000, high_stat*1000)
        elif group == 'prior':
            pop_title_str = ''
        
        #fig.suptitle(pop_title_str%(self.description[pop_stat], self.units[pop_stat], med_stat*1000, low_stat*1000, high_stat*1000), fontsize=20)
        fig.suptitle(pop_title_str, fontsize=20)
        fig.tight_layout()
        #if constant_ax:
        #    for ax in all_ax:
        #        ax.set_ylim(bottom=ax_min, top=ax_max)
        return fig

    def plot_all_studies(self, save_file=None, pop_stat='CLC', study_stat='CLC', pop=False, hero_unity=[], save=True):
        """Wrapper function for _study_ppc to plot all hero ids in a PDF

        Parameters
        ----------
        save_file : _type_, optional
            file name for pdf of fits. If None, use the default file name, by default None
        pop_stat : str, optional
            Population level PK statistic to display on plots, by default 'CLC'
        study_stat : str, optional
            Dataset level PK statistic to display on plots, by default 'CLC'
        pop : bool, optional
            _description_, by default False
        hero_unity : list, optional
            List of hero_ids to plot as unity plots instead of continuous time-course plots (better for individual animal data), by default []
        save : bool, optional
            _description_, by default True
        """
        continuous = True
        if self.model_type == '1-compartment':
            model_abrev = '1cmpt'
        elif self.model_type == '2-compartment':
            model_abrev = '2cmpt'
        if save_file is None:
            save_file = '%s_%s_%s_%s_%s.pdf'%(self.chemical, self.sex, self.species, pop_stat, model_abrev)
        # Test if species directory exists
        if not os.path.exists('../species_fits/%s'%self.species):
            os.mkdir('../species_fits/%s'%self.species)
        
        # Test if species/chemical path exists
        save_dir = '../species_fits/%s/%s'%(self.species, self.chemical)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        all_hero_ids = self.data.hero_id.unique()
        print('%s/%s'%(save_dir,save_file))
        with PdfPages('%s/%s'%(save_dir,save_file)) as pdf:
            for hero_id in all_hero_ids:
                if hero_id in hero_unity:
                    continuous = False
                fig = self._study_ppc(hero_id, pop_stat=pop_stat, study_stat=study_stat, continuous=continuous)

                if save:
                    pdf.savefig(fig, bbox_inches='tight')
                continuous = True

    def _build_ppc_data(self, tmp_df, ts=0, tf=None, t_length=1000, calc_BW=True):
        """Utility function that builds the data needed to run a time-course posterior predictive check
           tmp_df is the dataset-specific dataframe that has the dataset info
        """

        time = tmp_df[self.time_label]
        if tf is None:
            test_t = np.linspace(ts, 1.1*time.max(), t_length)
            #test_t = np.(ts, 1.1*time.max(), t_length)
        else:
            test_t = np.linspace(ts, tf, t_length)
        
        #test_y = np.empty(len(test_t))
        #test_Ni_summary = np.zeros(len(test_t), dtype=np.int32)
        #test_sd_summary = np.zeros(len(test_t), dtype=np.int32)
        test_route = [tmp_df[self.route_label].unique()[0]]*len(test_t) # Route of exposure for this dataset
        test_dose = [tmp_df[self.dose_label].unique()[0]]*len(test_t) # Applied dose for this dataset
        BW = tmp_df[self.BW_label] # Bodyweights used during the fits
        if calc_BW:
            BW_min = tmp_df.loc[tmp_df.time_cor==tmp_df.time_cor.min(), self.BW_label].mean()
            BW_max = tmp_df.loc[tmp_df.time_cor==tmp_df.time_cor.max(), self.BW_label].mean()
            #Interpolate the BW for all the new times in between the fitted times
            calc_BW = sci.interp1d(time.values, BW.values, bounds_error=False,
                                    fill_value=(BW_min, BW_max))
            test_BW = list(calc_BW(test_t))
        else:
            test_BW = [BW.mean()]*len(test_t)
        
        test_study_idx = [tmp_df['study_idx'].unique()[0]]*len(test_t)
        test_dataset_idx = [tmp_df['dataset_idx'].unique()[0]]*len(test_t)
        test_D_ss = [tmp_df[self.Dss_label].unique()[0]]*len(test_t)
        if self.model_structure == 'hierarchical':
            params = {'t': test_t,
                    #'y': test_y,
                    'route': test_route,
                    'D': test_dose,
                    'BW': test_BW,
                    'Dss': test_D_ss,
                    'sidx': test_study_idx,
                    'didx': test_dataset_idx}
        else:
            params = {'t': test_t,
                    #'y': test_y,
                    'route': test_route,
                    'D': test_dose,
                    'BW': test_BW,
                    'Dss': test_D_ss,
                    
                    }
        return params