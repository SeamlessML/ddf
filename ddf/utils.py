"""
DDF
Copyright (C) 2018 Seamless Global Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

"""

from collections import defaultdict

from six.moves import cPickle as pickle
import numpy as np


def ensure_is_list(obj):
    return obj if isinstance(obj, list) else [obj]


def ix_to_bool(ix, length):
    boolean_mask = np.repeat(False, length)
    boolean_mask[ix] = True
    return boolean_mask


def concatenate_dicts(data, fillna=None):
    all_keys = [d.keys() for d in data]
    flat_keys = [k for keys in all_keys for k in keys]
    keys = set(flat_keys)
    _data = {k: [] for k in keys}
    for row in data:
        for k in keys:
            _data[k].append(row.get(k, fillna))
    return _data


def is_npdatetime(v):
    try:
        answer = 'datetime' in v.dtype.name
    except:
        answer = False
    return answer


def is_datetime(v):
    return 'datetime' in str(v.dtype)


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def get_group_ixs(*group_ids, **kwargs):
    """ Returns a dictionary {groupby_id: group_ix}.

    group_ids:
        List of IDs to groupbyy
    kwargs:
        bools = True or False, if True returns a boolean array
    """
    group_ids = _ensure_group_ids_hashable(group_ids)
    grouped_ixs = _get_group_ixs(group_ids)
    grouped_ixs = _convert_int_indices_to_bool_indices_if_necessary(grouped_ixs, kwargs)
    return grouped_ixs


def _ensure_group_ids_hashable(group_ids):
    def is_list_of_list(ids):
        return isinstance(ids[0], list)

    def is_matrix(ids):
        return isinstance(ids, np.ndarray) and ids.ndim == 2

    if len(group_ids) == 1:
        combined_group_ids = group_ids[0]
    else:
        combined_group_ids = zip(*group_ids)

    if is_list_of_list(combined_group_ids) or is_matrix(combined_group_ids):
        hashable_group_ids = [tuple(group_id) for group_id in combined_group_ids]
    else:
        hashable_group_ids = combined_group_ids
    return hashable_group_ids


def _convert_int_indices_to_bool_indices_if_necessary(ixs, kwargs):
    bools = kwargs.get('bools', False)
    if bools:
        length = np.sum([len(v) for v in ixs.itervalues()])
        ixs = {k: ix_to_bool(v, length) for k, v in ixs.iteritems()}
    return ixs


def _get_group_ixs(ids):
    id_hash = defaultdict(list)
    for j, key in enumerate(ids):
        id_hash[key].append(j)
    id_hash = {k: np.array(v) for k, v in id_hash.iteritems()}
    return id_hash


def is_nptimedelta(v):
    try:
        answer = 'timedelta' in v.dtype.name
    except:
        answer = False
    return answer


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def nan_isclose(x, y):
    nan_ix_x = np.isnan(x)
    nan_ix_y = np.isnan(y)
    is_close = np.isclose(x, y)
    nan_close = (is_close | (nan_ix_x & nan_ix_y))
    return nan_close


def nan_allclose(x, y):
    return np.all(nan_isclose(x, y))
