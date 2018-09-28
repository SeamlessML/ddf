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

from collections import OrderedDict
import os

from ddf import DDF

import numpy as np


def thin_by_id(ddf, group_id, factor, offset=0):
    unique_groups = get_unique_values_in_order(ddf[group_id])
    keep_groups = unique_groups[offset::factor]
    include_ix = np.in1d(ddf[group_id], keep_groups)
    return ddf.rowslice(include_ix)


def get_unique_values_in_order(values):
    return list(OrderedDict.fromkeys(values))


def concatenate(dfs):
    new_dict = {}
    for col in dfs[0]:
        new_dict[col] = np.concatenate([df[col] for df in dfs])
    new_df = DDF(new_dict)
    return new_df


def append_csv(data, path):
    assert path.endswith('.csv')
    to_log = data if isinstance(data, DDF) else DDF(data)
    if os.path.isfile(path):
        current_log = DDF.from_csv(path)
        current_log = current_log.append(to_log, axis=0)
    else:
        current_log = to_log
    current_log.to_csv(path)
