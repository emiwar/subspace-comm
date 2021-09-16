import numpy as np

class GaussianProcessLayer:
    
    def __init__(self, period, sigma, dims):
        x = np.arange(period)
        dist_for = (period-x[:, np.newaxis] + x[np.newaxis, :])%period
        dist_back = (period+x[:, np.newaxis] - x[np.newaxis, :])%period
        dist = np.minimum(dist_for, dist_back)
        self.cov = np.exp(-(dist/sigma)**2)
        self.redraw(dims)
        
    def redraw(self, dims=None):
        '''Draw a new signal from the same covariance matrix'''
        if dims is None:
            dims = self.output.shape[1]
        self.output = np.random.multivariate_normal(np.zeros(self.cov.shape[0]),
                                                    self.cov, dims).T
        
    def get_output(self, t):
        '''Get the output of the process at time t'''
        return self.output[t%len(self.output), :]

class InterpolatedGPLayer(GaussianProcessLayer):
    
    def redraw(self, dims=None):
        self.output = np.random.multivariate_normal(np.zeros(self.cov.shape[0]),
                                                    self.cov, dims).T
        #self.output[:, :dims//2] /= np.sqrt(np.sum(self.output[:, :dims//2]**2, axis=1))[:, np.newaxis]
        #self.output[:,  dims//2:] /= np.sqrt(np.sum(self.output[:, dims//2:]**2, axis=1))[:, np.newaxis]
        period = self.output.shape[0]
        self.output[:, :dims//2] *= np.linspace(0, 1, period)[:, np.newaxis]
        self.output[:, dims//2:] *= (1-np.linspace(0, 1, period))[:, np.newaxis]