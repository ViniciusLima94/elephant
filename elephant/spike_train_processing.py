# -*- coding: utf-8 -*-
"""
Module for spike train processing

:copyright: Copyright 2014-2020 by the Elephant team, see `doc/authors.rst`.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division

import neo
import elephant.conversion as conv
import numpy as np


def _get_index(lst, obj):
    for index, item in enumerate(lst):
        if item is obj:
            return index


def _check_consistency_of_spiketrainlist(spiketrains, t_start=None,
                                         t_stop=None):
    if len(spiketrains) == 0:
        raise ValueError('The spiketrains should not be empty!')
    for spiketrain in spiketrains:
        if not isinstance(spiketrain, neo.SpikeTrain):
            raise TypeError(
                "spike train must be instance of :class:`SpikeTrain` of Neo!\n"
                "    Found: %s, value %s" % (
                    type(spiketrain), str(spiketrain)))
        if t_start is None and not spiketrain.t_start == spiketrains[
                0].t_start:
            raise ValueError(
                "the spike trains must have the same t_start!")
        if t_stop is None and not spiketrain.t_stop == spiketrains[
                0].t_stop:
            raise ValueError(
                "the spike trains must have the same t_stop!")


def detect_synchrofacts(spiketrains, sampling_rate, spread=1,
                        deletion_threshold=None, invert_delete=False):
    """
    Given a list of neo.Spiketrain objects, calculate the number of synchronous
    spikes found and optionally delete or extract them from the given list
    *in-place*.

    The spike trains are binned at sampling precission
    (i.e. bin_size = 1 / `sampling_rate`)

    Two spikes are considered synchronous if they occur separated by strictly
    fewer than `spread - 1` empty bins from one another. See
    `elephant.statistics.complexity_intervals` for a detailed description of
    how synchronous events are counted.

    Synchronous events are considered within the same spike train and across
    different spike trains in the `spiketrains` list. Such that, synchronous
    events can be found both in multi-unit and single-unit spike trains.

    The spike trains in the `spiketrains` list are annotated with the
    complexity value of each spike in their :attr:`array_annotations`.


    Parameters
    ----------
    spiketrains : list of neo.SpikeTrains
        a list of neo.SpikeTrains objects. These spike trains should have been
        recorded simultaneously.
    sampling_rate : pq.Quantity
        Sampling rate of the spike trains. The spike trains are binned with
        bin_size = 1 / `sampling_rate`.
    spread : int
        Number of bins in which to check for synchronous spikes.
        Spikes that occur separated by `spread - 1` or less empty bins are
        considered synchronous.
        Default: 1
    deletion_threshold : int, optional
        Threshold value for the deletion of spikes engaged in synchronous
        activity.
          * `deletion_threshold = None` leads to no spikes being deleted, spike
            trains are array-annotated and the spike times are kept unchanged.
          * `deletion_threshold >= 2` leads to all spikes with a larger or
            equal complexity value to be deleted *in-place*.
          * `deletion_threshold` cannot be set to 1 (this would delete all
            spikes and there are definitely more efficient ways of doing this)
          * `deletion_threshold <= 0` leads to a ValueError.
        Default: None
    invert_delete : bool
        Inversion of the mask for deletion of synchronous events.
          * `invert_delete = False` leads to the deletion of all spikes with
            complexity >= `deletion_threshold`,
            i.e. deletes synchronous spikes.
          * `invert_delete = True` leads to the deletion of all spikes with
            complexity < `deletion_threshold`, i.e. returns synchronous spikes.
        Default: False

    Returns
    -------
    complexity_epoch : neo.Epoch
        An epoch object containing complexity values, left edges and durations
        of all intervals with at least one spike.

        Calculated with
        `elephant.spike_train_processing.precise_complexity_intervals`.

        Complexity values per spike can be accessed with:

        >>> complexity_epoch.array_annotations['complexity']

        The left edges of the intervals with:

        >>> complexity_epoch.times

        And the durations with:

        >>> complexity_epoch.durations

    See also
    --------
    elephant.statistics.precise_complexity_intervals

    """
    if deletion_threshold is not None and deletion_threshold <= 1:
        raise ValueError('A deletion_threshold <= 1 would result'
                         'in deletion of all spikes.')

    # find times of synchrony of size >=n
    complexity_epoch = precise_complexity_intervals(spiketrains,
                                                    sampling_rate,
                                                    spread=spread)
    complexity = complexity_epoch.array_annotations['complexity']
    right_edges = complexity_epoch.times + complexity_epoch.durations

    # j = index of pre-selected sts in spiketrains
    # idx = index of pre-selected sts in original
    # block.segments[seg].spiketrains
    for idx, st in enumerate(spiketrains):

        # all indices of spikes that are within the half-open intervals
        # defined by the boundaries
        # note that every second entry in boundaries is an upper boundary
        spike_to_epoch_idx = np.searchsorted(right_edges,
                                             st.times.rescale(
                                                 right_edges.units))
        complexity_per_spike = complexity[spike_to_epoch_idx]

        st.array_annotate(complexity=complexity_per_spike)

        if deletion_threshold is not None:
            mask = complexity_per_spike < deletion_threshold
            if invert_delete:
                mask = np.invert(mask)
            old_st = st
            new_st = old_st[mask]
            spiketrains[idx] = new_st
            unit = old_st.unit
            segment = old_st.segment
            if unit is not None:
                unit.spiketrains[_get_index(unit.spiketrains,
                                            old_st)] = new_st
            if segment is not None:
                segment.spiketrains[_get_index(segment.spiketrains,
                                               old_st)] = new_st
            del old_st

    return complexity_epoch


def precise_complexity_intervals(spiketrains, sampling_rate, spread=0):
    """
    Calculate the complexity (i.e. number of synchronous spikes found)
    at `sampling_rate` precision in a list of spiketrains.

    Complexity is calculated by counting the number of spikes (i.e. non-empty
    bins) that occur separated by `spread - 1` or less empty bins, within and
    across spike trains in the `spiketrains` list.


    Parameters
    ----------
    spiketrains : list of neo.SpikeTrains
        a list of neo.SpikeTrains objects. These spike trains should have been
        recorded simultaneously.
    sampling_rate : pq.Quantity
        Sampling rate of the spike trains.
    spread : int, optional
        Number of bins in which to check for synchronous spikes.
        Spikes that occur separated by `spread - 1` or less empty bins are
        considered synchronous.
          * `spread = 0` corresponds to a bincount accross spike trains.
          * `spread = 1` corresponds to counting consecutive spikes.
          * `spread = 2` corresponds to counting consecutive spikes and
            spikes separated by exactly 1 empty bin.
          * `spread = n` corresponds to counting spikes separated by exactly
            or less than `n - 1` empty bins.
        Default: 0

    Returns
    -------
    complexity_intervals : neo.Epoch
        An epoch object containing complexity values, left edges and durations
        of all intervals with at least one spike.

    Raises
    ------
    ValueError
        When `t_stop` is smaller than `t_start`.

    Examples
    --------
    Here the behavior of
    `elephant.spike_train_processing.precise_complexity_intervals` is shown, by
    applying the function to some sample spiketrains.

    >>> import neo
    >>> import quantities as pq

    >>> sampling_rate = 1/pq.ms
    >>> st1 = neo.SpikeTrain([1, 4, 6] * pq.ms, t_stop=10.0 * pq.ms)
    >>> st2 = neo.SpikeTrain([1, 5, 8] * pq.ms, t_stop=10.0 * pq.ms)

    >>> # spread = 0, a simple bincount
    >>> ep1 = precise_complexity_intervals([st1, st2], sampling_rate)
    >>> print(ep1.array_annotations['complexity'].flatten())
    [2 1 1 1 1]
    >>> print(ep1.times)
    [0.  3.5 4.5 5.5 7.5] ms
    >>> print(ep1.durations)
    [1.5 1.  1.  1.  1. ] ms

    >>> # spread = 1, consecutive spikes
    >>> ep2 = precise_complexity_intervals([st1, st2], sampling_rate, spread=1)
    >>> print(ep2.array_annotations['complexity'].flatten())
    [2 3 1]
    >>> print(ep2.times)
    [0.  3.5 7.5] ms
    >>> print(ep2.durations)
    [1.5 3.  1. ] ms

    >>> # spread = 2, consecutive spikes and separated by 1 empty bin
    >>> ep3 = precise_complexity_intervals([st1, st2], sampling_rate, spread=2)
    >>> print(ep3.array_annotations['complexity'].flatten())
    [2 4]
    >>> print(ep3.times)
    [0.  3.5] ms
    >>> print(ep3.durations)
    [1.5 5. ] ms

    """
    _check_consistency_of_spiketrainlist(spiketrains)

    bin_size = 1 / sampling_rate

    bst = conv.BinnedSpikeTrain(spiketrains,
                                binsize=bin_size)

    bincount = np.array(bst.to_sparse_array().sum(axis=0)).squeeze()

    if spread == 0:
        bin_indices = np.nonzero(bincount)[0]
        complexities = bincount[bin_indices]
        left_edges = bst.bin_edges[bin_indices]
        right_edges = bst.bin_edges[bin_indices + 1]
    else:
        i = 0
        complexities = []
        left_edges = []
        right_edges = []
        while i < len(bincount):
            current_bincount = bincount[i]
            if current_bincount == 0:
                i += 1
            else:
                last_window_sum = current_bincount
                last_nonzero_index = 0
                current_window = bincount[i:i + spread + 1]
                window_sum = current_window.sum()
                while window_sum > last_window_sum:
                    last_nonzero_index = np.nonzero(current_window)[0][-1]
                    current_window = bincount[i:
                                              i + last_nonzero_index
                                              + spread + 1]
                    last_window_sum = window_sum
                    window_sum = current_window.sum()
                complexities.append(window_sum)
                left_edges.append(
                    bst.bin_edges[i].magnitude.item())
                right_edges.append(
                    bst.bin_edges[
                        i + last_nonzero_index + 1
                    ].magnitude.item())
                i += last_nonzero_index + 1

        # we dropped units above, because neither concatenate nor append works
        # with arrays of quantities
        left_edges *= bst.bin_edges.units
        right_edges *= bst.bin_edges.units

    # ensure that spikes are not on the bin edges
    bin_shift = .5 / sampling_rate
    left_edges -= bin_shift
    right_edges -= bin_shift

    # ensure that an epoch does not start before the minimum t_start
    min_t_start = min([st.t_start for st in spiketrains])
    left_edges[0] = min(min_t_start, left_edges[0])

    complexity_epoch = neo.Epoch(times=left_edges,
                                 durations=right_edges - left_edges,
                                 array_annotations={'complexity':
                                                    complexities})

    return complexity_epoch
