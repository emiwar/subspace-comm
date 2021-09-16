import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import figurefirst
import tqdm

from layers.efficient_coding import EfficientCodingLayer
from layers.gaussian_process import GaussianProcessLayer, InterpolatedGPLayer

class FeedforwardSplit:
    
    def __init__(self, n_neurons, dims=[2, 2]):
        tot_dims = sum(dims)
        #self.input_trajectory = GaussianProcessLayer(2000, 200, tot_dims)
        self.input_trajectory = InterpolatedGPLayer(2000, 200, tot_dims)
        self.first_layer = EfficientCodingLayer(n_neurons, tot_dims)
        self.output_layers = []
        for d in dims:
            output_layer = EfficientCodingLayer(n_neurons, d)
            self.output_layers.append(output_layer)
        self.t = 0.0
        self.spike_log = pd.DataFrame(columns=["layer", "time", "id"])
        self.readout_log = {'input': [], 'sender': [],
                            'receiver0': [], 'receiver1': []}
        
    def log_spikes(self, pop, spikes):
        self.spike_log = self.spike_log.append(pd.DataFrame({"layer": pop,
                                            "time": self.t,
                                            "id": spikes}))
    
    def step(self):
        inp = 5*self.input_trajectory.get_output(int(self.t))
        self.readout_log['input'].append(inp)
        spikes = self.first_layer.step(inp)
        self.log_spikes("sender", spikes)
        self.readout_log['sender'].append(self.first_layer.get_output())
        ind = 0
        for i, output_layer in enumerate(self.output_layers):
            d = output_layer.gamma.shape[1]
            layer_input = self.first_layer.get_output()[ind:ind+d]
            spikes = output_layer.step(layer_input)
            self.log_spikes("receiver{}".format(i), spikes)
            self.readout_log["receiver{}".format(i)].append(output_layer.get_output())
            ind += d
        self.t += 1

net = FeedforwardSplit(200)
for t in tqdm.trange(2000):
    net.step()

layout = figurefirst.FigureLayout("subspace_comm_template.svg")
layout.make_mplfigures()
for r in ("input", "receiver0", "receiver1"):
    ax = layout.axes["{}_readout".format(r)]["axis"]
    if r=="receiver1":
        #Hack to move the color cykler
        ax.plot([],[])
        ax.plot([],[])
    ax.plot(net.readout_log[r])
    ax.axhline(0, color="k", alpha=.2)
    ax.axis("off")
    
for p in ("sender", "receiver0", "receiver1"):
    ax = layout.axes["{}_raster".format(p)]["axis"]
    spikes = net.spike_log[net.spike_log.layer == p]
    ax.plot(spikes.time, spikes.id, 'k.', ms=1)
    sns.despine(ax=ax, bottom=True, trim=True)
    ax.set_xticks([])
    
    binned = spikes.groupby((spikes.time/50).astype(np.int)).size()
    ax = layout.axes["{}_bins".format(p)]["axis"]
    ax.bar(binned.index*50, binned/200*(50.0), width=50, color="k", align="edge")
    sns.despine(ax=ax)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_ylim(0, 41)
    ax.set_yticks([0,20,40])
layout.axes["input_readout"]["axis"].set_title("Input manifold\ntrajectory (4D)")
layout.axes["sender_raster"]["axis"].set_title("Sender pop")
layout.axes["receiver0_raster"]["axis"].set_title("Receiver pop 1")
layout.axes["receiver1_raster"]["axis"].set_title("Receiver pop 2")
layout.axes["receiver0_readout"]["axis"].set_title("Reconstructed\ntrajectory (2D)")
layout.axes["receiver1_readout"]["axis"].set_title("Reconstructed\ntrajectory (2D)")
layout.insert_figures('plots')
layout.write_svg("subspace_comm_linear_interpolation.svg")
