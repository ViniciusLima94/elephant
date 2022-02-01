from copy import deepcopy

import numpy as np
import quantities as pq

from elephant.statistics import isi


class MeanOnline(object):
    def __init__(self, batch_mode=False):
        self.mean = None
        self.count = 0
        self.units = None
        self.batch_mode = batch_mode

    def update(self, new_val):
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.batch_mode:
            batch_size = new_val.shape[0]
            new_val_sum = new_val.sum(axis=0)
        else:
            batch_size = 1
            new_val_sum = new_val
        self.count += batch_size
        if self.mean is None:
            self.mean = deepcopy(new_val_sum / batch_size)
            self.units = units
        else:
            if units != self.units:
                raise ValueError("Each batch must have the same units.")
            self.mean += (new_val_sum - self.mean * batch_size) / self.count

    def as_units(self, val):
        if self.units is None:
            return val
        return pq.Quantity(val, units=self.units, copy=False)

    def get_mean(self):
        return self.as_units(deepcopy(self.mean))

    def reset(self):
        self.mean = None
        self.count = 0
        self.units = None


class VarianceOnline(MeanOnline):
    def __init__(self, batch_mode=False):
        super(VarianceOnline, self).__init__(batch_mode=batch_mode)
        self.variance_sum = 0.

    def update(self, new_val):
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.mean is None:
            self.mean = 0.
            self.variance_sum = 0.
            self.units = units
        elif units != self.units:
            raise ValueError("Each batch must have the same units.")
        delta_var = new_val - self.mean
        if self.batch_mode:
            batch_size = new_val.shape[0]
            self.count += batch_size
            delta_mean = new_val.sum(axis=0) - self.mean * batch_size
            self.mean += delta_mean / self.count
            delta_var *= new_val - self.mean
            delta_var = delta_var.sum(axis=0)
        else:
            self.count += 1
            self.mean += delta_var / self.count
            delta_var *= new_val - self.mean
        self.variance_sum += delta_var

    def get_mean_std(self, unbiased=False):
        if self.mean is None:
            return None, None
        if self.count > 1:
            count = self.count - 1 if unbiased else self.count
            std = np.sqrt(self.variance_sum / count)
        else:
            # with 1 update biased & unbiased sample variance is zero
            std = 0.
        mean = self.as_units(deepcopy(self.mean))
        std = self.as_units(std)
        return mean, std

    def reset(self):
        super(VarianceOnline, self).reset()
        self.variance_sum = 0.


class InterSpikeIntervalOnline(object):
    def __init__(self, bin_size=0.0005, max_isi_value=1, batch_mode=False):
        self.max_isi_value = max_isi_value  # in sec
        self.last_spike_time = None
        self.bin_size = bin_size  # in sec
        self.num_bins = int(self.max_isi_value / self.bin_size)
        self.bin_edges = np.linspace(start=0, stop=self.max_isi_value,
                                     num=self.num_bins + 1)
        self.current_isi_histogram = np.zeros(shape=self.num_bins)
        self.bach_mode = batch_mode
        self.units = None

    def update(self, new_val):
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.last_spike_time is None:  # for first batch
            if self.bach_mode:
                new_isi = isi(new_val)
                self.last_spike_time = new_val[-1]
            else:
                new_isi = np.array([])
                self.last_spike_time = new_val
            self.units = units
        else:  # for second to last batch
            if units != self.units:
                raise ValueError("Each batch must have the same units.")
            if self.bach_mode:
                new_isi = isi(np.append(self.last_spike_time, new_val))
                self.last_spike_time = new_val[-1]
            else:
                new_isi = np.array([new_val - self.last_spike_time])
                self.last_spike_time = new_val
        isi_hist, _ = np.histogram(new_isi, bins=self.bin_edges)
        self.current_isi_histogram += isi_hist

    def as_units(self, val):
        if self.units is None:
            return val
        return pq.Quantity(val, units=self.units, copy=False)

    def get_isi(self):
        return self.as_units(deepcopy(self.current_isi_histogram))

    def reset(self):
        self.last_spike_time = None
        self.units = None
        self.current_isi_histogram = np.zeros(shape=self.num_bins)
