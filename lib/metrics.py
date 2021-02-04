import numpy as np
import torch


class Metrics():
    def __init__(self):
        self.reset()

    def reset(self):
        self.running_agg = {}
        self.running_count = {}
        self.hists = {}

    @torch.no_grad()
    def step(self, **additional_terms):
        for term in additional_terms:
            if term not in self.running_agg:
                self.running_agg[term] = additional_terms[term].detach()
                self.running_count[term] = 1
            else:
                self.running_agg[term] += additional_terms[term].detach()
                self.running_count[term] += 1


    def step_hist(self, **data):
        for term in data:
            vals = data[term].detach()
            if term not in self.hists:
                self.hists[term] = [vals]
            else:
                self.hists[term].append(vals)


    @torch.no_grad()
    def evaluate(self):
        values = {}
        for key in self.running_agg:
            values[key] = float(self.running_agg[key] / self.running_count[key])
        hists = {}
        for key in self.hists:
            hist_vals = torch.cat(self.hists[key]).cpu().numpy()
            low  = hist_vals.min() - 0.5
            high = hist_vals.max() + 1.0
            hist = np.histogram(hist_vals, bins=np.arange(low, high))
            hists[key] = hist
        self.reset()
        return values, hists

