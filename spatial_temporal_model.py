import george
import pandas as pd
from gpmc import GPMC

COLUMNS = ['region_id', 'log_crime_rate', 'x1', 'x2', 'centroid_x', 'centroid_y', 'time']

class SpatialTempGP(GPMC):
    """
    All methods for calculating GP as well as its prior/likelihood and posterior.
    The better way is to have a seperate class instead of inheritating from `GPMC` as 
    GP class does not interact with its parent class; however as the used methods 
    (i.e., `kernel_gp` and `lnprob_gp`) are not static, configuring and initilization of 
    GPMC class is ugly.
    """
    def __spatial_kernel_gp(self, kernelname):
        return super(self.__class__, self).kernel_gp(kernelname=kernelname)

    def __temporal_kernel_gp(self, kernelname):
        """ Kernel defininition
        Standard kernels provided with george, evtl replaced with manual customizable kernels
        User can change function to add new kernels or comninations thereof as much as required 
        :param kernelname: name of kernel, either 'expsquared', 'matern32', or 'rationalq'
        """
        if kernelname == 'expsine':
            kernel = 1. * george.kernels.ExpSine2Kernel(gamma=0.1, period=12, ndim = 2)
        if kernelname == 'cosine':
            kernel = 1. * george.kernels.CosineKernel(1., ndim = 2)
        return kernel
    
    def kernel_gp(self, kernelname='expsquared', **kwargs):
        """
        GP kernel have two parts: one for spatial correlation and another for temporal.
        Temporal corellation is likely to be seasonal and periodic on hour/day/week/year.
        It requires a close look at the time to see if any hourly/daily/weekly/yearly pattern
        is observed.
        :param kernelname:
        :param kwargs:
        :return:
        """
        temporal_kernel_name = kwargs['temporal_kernel_name']
        spatial_kernel = self.__spatial_kernel_gp(kernelname=kernelname)
        temporal_kernel = self.__spatial_kernel_gp(kernelname=temporal_kernel_name)
        return spatial_kernel + temporal_kernel
    
    def __lnprior_gp(self, prior_gp=1):
        """ 
        The prior for GP; by default it is a flat prior 
        """
        return np.log(prior_gp)
    
    def __lnlikelihood_gp(self, p, sigma, dropout):
        """
        The likelihood of the GP; since the prior in the parent class assumes a flat prior, so the
        output of `lnprob_gp()` is actually the GP likelihood
        """
        return super(self.__class__, self).lnprob_gp(
            p=p,
            sigma=sigma,
            dropout=dropout)    
    
    def lnprob_gp(self, p, sigma, dropout=0.):
        """ Combines prior and likelihood of GP to create GP posterior """
        return self.__lnprior_gp() + self.__lnlikelihood_gp(p=p, 
                                                            sigma=sigma,
                                                            dropout=dropout)




class SpatialTempBLR(GPMC):
    def __init__(self, t):
        super(self.__class__, self).__init__(
            outmcmc=outmcmc, split_traintest=split_traintest
        )
        t = t.reshape(-1,1)
        assert 1 not in t.shape
        
        self.t = t.T if t.shape[1] != 1  else t # `t` is a N*1
        
    def __concat_time_and_beta(self, beta):
        return np.insers(beta, 0, self.t)
    
    def lnprior_blr(self, beta, sigma):
        """
        beta must include `t` as well. So, `t` must be added.
        `t` is a N*1 list and `beta` is N*M matrix. The concatenation
        will be of N*(M+1)
        """
        return super(self.__class__, self).lnprior_blr(
            beta=self.__concat_time_and_beta(beta=beta),
            sigma=sigma
        )

    def lnlikelihood_blr(self, alpha, beta, sigma):
        """
        beta must include `t` as well. So, `t` must be added.
        """
        return super(self.__class__, self).lnlikelihood_blr(
            alpha=alpha,
            beta=self.__concat_time_and_beta(beta=beta),
            sigma=sigma
        )
    
    def lnprob_blr(self, alpha, beta, sigma):
        #TODO: not necessary to have this function. Just for visiblity
        return return super(self.__class__, self).lnprob_blr(
            alpha=alpha, 
            beta=beta, 
            sigma=sigma)


class MCMC(GPMC):
    pass

class Plotting(GPMC):
    pass

class DataProcessing(GPMC):
    pass


class SpatialTemporalModel(SpatialTempGP, SpatialTempBLR, MCMC, Plotting, DataProcessing):
    """
    This implementation is based on the paper " " which models the spatial-temporal phenomena as
    as log Cox (as an extension of Poisson distribution).
    This class uses the pre implemented functions in GPMC class modifying the necessary parts. Similar to
    GPMC class, emcee package is used for MCMC sampling.
    """
    def __init__(self, outmcmc, split_traintest):
        super(self.__class__, self).__init__(
            outmcmc=outmcmc, split_traintest=split_traintest
        )
        self.dataframe = pd.DataFrame(self.data, columns=COLUMNS)
        self.dataframe['time'] = pd.to_datetime(self.dataframe['time'], dayfirst=True, infer_datetime_format=True)
        self.dataframe['week'] = self.dataframe.resample('W')

    def population_density_factor(self):
        """ 
        calculate population density in place `s` at time `t` as λ(s,t) in Eq.1
        """
        λ = self.dataframe.sort_values(['region_id', 'week'])
        return λ[['region_id', 'week']].values

    def crime_risk_factor(self, data):
        """ calculate crime risk at place `t` at time `t` as R(s,t) in Eq.1 """
        return np.lo

    def log_cox(self, data):
        """
        Creates the linear regression as
        b0 + b1*t + b2*x1 + ... + bn*xn
        """
        pass




