import numpy as np

class EfficientCodingLayer:
    
    def __init__(self, n_neurons, n_dims, tau_mem=20.0, mu=0.1, nu=0.1):
        self.gamma = np.random.normal(0, 1, (n_neurons, n_dims))
        self.gamma /= np.sqrt(np.sum(self.gamma**2, axis=1))[:, np.newaxis]
        self.tau_mem = tau_mem
        self.v = np.zeros(n_neurons)
        self.filtered_spikes = np.zeros(n_neurons)
        self.thres = 0.5*(nu/tau_mem + mu/(tau_mem**2)+1)
        self.mu = mu
        
    def step(self, manifold_current, dt=1.0):
        if manifold_current.shape[0] != self.gamma.shape[1]:
            raise ValueError("The manifold current should be specified per subspace dimension.")
        
        #Gamma is the projection from the manifold to the neurons
        inp_current = np.dot(self.gamma, manifold_current)
        
        #Leaky integrator dynamics
        self.v = (1 - dt/self.tau_mem)*self.v + (dt/self.tau_mem)*inp_current
        
        #The filter of the spikes of the neurons is leaky
        self.filtered_spikes = (1 - dt/self.tau_mem)*self.filtered_spikes
        
        #List of all the neurons that spiked in this timestep
        spiking_neurons = []
        
        #Rapid recurrent connections. This is a bit tricky numerically:
        #if two neurons reach the threshold at the same time, we can't say that both
        #spiked, because if the first neuron spiked eps ms before the second, where
        #eps<<dt, it's rapid synapses might inhibit the second one before it reaches
        #the threshold. This can either be solved with a very small dt, or as here,
        #by checking the threshold for one (random) neuron at a time, do it's rapid
        #synapses, and then check thresholds again.
        while True:
            above_threshold = np.nonzero(self.v > self.thres)[0]
            
            if above_threshold.size==0:
                break
            
            #Pick a random neuron above the threshold
            spiker = np.random.choice(above_threshold)
            
            #Add the ID of the neuron that spiked
            spiking_neurons.append(spiker)
            
            #Add the spikes to the filtered output
            self.filtered_spikes[spiker] += 1          
            
            #Reset the spiking neurons
            self.v[spiker] -= self.mu/(self.tau_mem**2)
            
            #Calculate the (delta spike) currents from the recurrent connections
            rec_currents = -np.dot(self.gamma, self.gamma[spiker, :])
            
            #Add them to the voltage.
            self.v += rec_currents
            
        #Convert list into one array
        spiking_neurons = np.array(spiking_neurons)
        
        return spiking_neurons
    
    def get_output(self, t=None):
        return np.dot(self.gamma.T, self.filtered_spikes)
    
