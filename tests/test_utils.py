import numpy as np
import pytest

from ddf import utils


test_data = [(1, [1]), ([], []), ([1, 2, 3], [1, 2, 3])]


@pytest.mark.parametrize('argument, expected', test_data)
def test_ensure_is_list(argument, expected):
    out = utils.ensure_is_list(argument)
    assert out == expected


def test_ix_to_bool():
    rows = [2, 3, 4]
    bools = utils.ix_to_bool(rows, 10)
    assert not any(bools[0:2])
    assert not any(bools[5:])
    assert all(bools[2:5])
    assert len(bools) == 10


def test_concatenate_dicts():
    dicts = [
            {'a': 1, 'b': 2},
            {'a': 3, 'b': 2, 'c': 3},
            ]
    conc = utils.concatenate_dicts(dicts)
    expected = {'a': [1, 3], 'b': [2, 2], 'c': [None, 3]}
    assert conc == expected


def test_is_npdatetime():
    v = np.array(['2001-01-01', 'NaT'], dtype='datetime64[ns]')
    assert utils.is_npdatetime(v)


def test_get_group_ixs_list_of_ids():
    group_ids = [0, 1, 0, 2]
    ixs = utils.get_group_ixs(group_ids)
    expected_ix = {0: [0, 2], 1: [1], 2: [3]}
    assert_dicts_are_equal(ixs, expected_ix)


def test_get_group_ixs_vector_of_ids():
    group_ids = np.array([0, 1, 0, 1])
    ixs = utils.get_group_ixs(group_ids)
    expected_ix = {0: [0, 2], 1: [1, 3]}
    assert_dicts_are_equal(ixs, expected_ix)


def test_get_group_ixs_multiple_id_args():
    group0_ids = [0, 1, 2, 0]
    group1_ids = [1, 0, 2, 1]
    ixs = utils.get_group_ixs(group0_ids, group1_ids)
    expected_ix = {(0, 1): [0, 3], (1, 0): [1], (2, 2): [2]}
    assert_dicts_are_equal(ixs, expected_ix)


def test_get_group_ixs_list_of_list_of_ids():
    group_ids = [[0, 1], [1, 0], [1, 1], [1, 0]]
    ixs = utils.get_group_ixs(group_ids)
    expected_ix = {(0, 1): [0], (1, 0): [1, 3], (1, 1): [2]}
    assert_dicts_are_equal(ixs, expected_ix)


def test_get_group_ixs_matrix_of_ids():
    group_ids = np.array([[0, 1], [1, 0], [1, 0], [1, 0]])
    ixs = utils.get_group_ixs(group_ids)
    expected_ix = {(0, 1): [0], (1, 0): [1, 2, 3]}
    assert_dicts_are_equal(ixs, expected_ix)


def test_get_group_ixs_conversion_to_bools():
    group_ids = range(4)
    ixs = utils.get_group_ixs(group_ids, bools=True)
    expected_ix = {
            0: np.array([True, False, False, False]),
            1: np.array([False, True, False, False]),
            2: np.array([False, False, True, False]),
            3: np.array([False, False, False, True]),
            }
    assert_dicts_are_equal(ixs, expected_ix)


def assert_dicts_are_equal(a, b):
    keys = set(a.keys() + b.keys())
    for k in keys:
        assert (a[k] == b[k]).all()


def test_nan_allclose():
    x = np.array([np.nan, 1])
    x = np.vstack([x, x]).T
    y = np.array([np.nan, 1-1e-12])
    y = np.vstack([y, y]).T
    assert utils.nan_allclose(x, y)
