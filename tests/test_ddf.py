from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from ddf import DDF, utils

np.random.seed(123)
N_EXAMPLES = 20

DATA = {
        0: np.random.randn(N_EXAMPLES),
        1: np.arange(N_EXAMPLES),
        2: np.random.choice([1, 2, np.nan], size=N_EXAMPLES),
        'group_id': np.random.choice([1, 2, 3], size=N_EXAMPLES)
        }


@pytest.fixture
def pandas_df():
    pdf = pd.DataFrame(DATA)
    return pdf


@pytest.fixture
def df():
    df = DDF(DATA)
    return df


def test_rename_ignore_missing(df):
    mapping = {'missing_column': 5, 'group_id': 'new_name'}
    df = df.rename(mapping, ignore_missing=True)
    assert 'new_name' in df
    assert 'group_id' not in df


def test_can_pickle(df, tmpdir):
    df['strings'] = np.repeat('a', len(df))
    path = str(tmpdir) + 'df.pkl'
    utils.write_pickle(df, path)
    loaded = utils.read_pickle(path)
    assert loaded.equals(df)
    for col in df:
        assert loaded[col].dtype == df[col].dtype


# TODO: make repr work for this
# df = DDF({'badcol': [u'', u'  ', u'+ ', u'\xa7', u'x ', u'xx', u'd ', u'\xa7\xa7', u'p ', u'? ']})

def test_ddf_repr():
    df = DDF()
    repr(df)


def test_ddf_init():
    data = [{'a': 1}]
    df = DDF(data)
    assert isinstance(df.data['a'], np.ndarray)


def test_fillna_not_inplace(df):
    df[0][:5] = np.nan
    original_df = df.copy(deep=True)
    filler = 999
    new_df = df.fillna(filler)
    assert np.allclose(new_df[0][:5], filler)
    assert np.array_equal(new_df[0][5:], original_df[0][5:])
    assert utils.nan_allclose(df[0], original_df[0])

    filled = df.fillna({0: 101, 2: 999})
    assert np.allclose(filled[0][:5], 101)
    assert np.allclose(filled[2][-4], 999)

    filled = df.fillna({0: np.arange(len(df))})
    assert np.array_equal(filled[0][:5], np.arange(5))


def test_fillna_inplace(df):
    df[0][:5] = np.nan
    original_df = df.copy(deep=True)
    filler = 999
    df.fillna(filler, inplace=True)
    assert np.allclose(df[0][:5], filler)
    assert np.array_equal(df[0][5:], original_df[0][5:])


def test_add_column_which_is_shape_n_1():
    DDF({'col': np.arange(10), 'new_col': np.arange(10).reshape((-1, 1))})


def test_saving_to_csv_by_default_does_not_save_index(tmpdir):
    save_path = str(tmpdir) + '/saved_df.csv'
    df = DDF({'col': np.arange(5)})
    df.to_csv(save_path)
    loaded_df = DDF.from_csv(save_path)
    assert 'Unnamed: 0' not in loaded_df


def test_add_column_to_empty_ddf():
    df = DDF()
    df['a'] = np.arange(5)


def test_set_single_col_with_matrix(df):
    zeros = np.zeros(len(df)).reshape((-1, 1))
    df['a'] = zeros
    assert len(df['a']) == len(df)
    assert np.array_equal(df['a'], zeros.reshape((-1, )))


def test_set_single_col_with_array(df):
    zeros = np.zeros(len(df))
    df['a'] = zeros
    assert len(df['a']) == len(df)
    assert np.allclose(df['a'], zeros)


def test_set_single_col_with_scalar(df):
    df['a'] = 0
    assert len(df['a']) == len(df)
    assert np.allclose(df['a'], 0)


def test_set_multiple_cols_to_same_scalar(df):
    df[['a', 'b']] = 0
    for col in 'a', 'b':
        values = df[col]
        assert np.array_equal(values, np.zeros(len(df)))


def test_set_multiple_existing_cols_to_different_scalars(df):
    cols = df.columns[:2]
    df[cols] = [0, 1]
    assert np.array_equal(df[cols[0]], np.zeros(len(df)))
    assert np.array_equal(df[cols[1]], np.ones(len(df)))


def test_set_multiple_cols_to_matrix(df):
    df[['a', 'b']] = np.zeros((len(df), 2))
    for col in 'a', 'b':
        assert np.array_equal(df[col], np.zeros(len(df)))


def test_adding_columns_to_df_with_one_scalar_row():
    df = DDF({'a': 0.10229005573016618})
    df['b'] = 1


def test_appropriate_setitem(df):
    before = df[1]
    df[1] = np.random.uniform(size=N_EXAMPLES)
    after = df[1]
    assert np.all(before != after)


def test_incorrect_length_setitem(df):
    with pytest.raises(AssertionError):
        df[1] = np.random.randn(10 * N_EXAMPLES)

##########################################
# # __init__
##########################################


def test_empty_init():
    DDF()
    DDF({})


def test_init_with_different_value_shapes():
    data = {'a': 1, 'b': np.arange(2)}
    with pytest.raises(RuntimeWarning):
        DDF(data)


def test_init_with_matrix_values():
    data = {'a': np.arange(10).reshape(5, 2)}
    with pytest.raises(RuntimeWarning):
        DDF(data)

##########################################
# # side effects
##########################################


def test_for_side_effects_outside_df():
    data = {'a': np.arange(5)}
    df = DDF(data)
    data['a'] += 1
    assert np.array_equal(df['a'], np.arange(5))


def test_for_side_effects_in_init_data():
    data = {'a': np.arange(5)}
    data['b'] = data['a']
    df = DDF(data)
    df['a'] += 1
    assert np.array_equal(df['b'], np.arange(5))


def test_for_side_effects_in_added_column():
    df = DDF({})
    df['a'] = np.arange(5)
    df['b'] = df['a']
    df['a'] += 1
    assert np.array_equal(df['b'], np.arange(5))

##########################################
# # Appending
##########################################


def test_appending_dataframes(df):
    new_df = df.append(df)
    assert len(new_df) == 2 * len(df)
    assert new_df.columns == df.columns

    new_col = df.colslice(0)
    new_col.rename({0: 99}, inplace=True)
    new_df = df.append(new_col)
    assert len(new_df) == len(df)
    assert 99 in new_df


def test_appending_with_timedeltas():
    df = DDF({'delta': np.array([100, 'NaT'], dtype='timedelta64[ns]')})
    other_df = DDF()
    df.append(other_df, axis=0)


def test_appending_dfs_with_one_scalar_row():
    df = DDF({'a': 0.10229005573016618})
    other_df = DDF({'b': 0.10229005573016618})
    df.append(other_df, axis=0)


def test_appending_new_df_with_new_cols():
    d1 = DDF({'col1': np.arange(4), 'col2': np.arange(4)})
    d2 = DDF({'col1': np.arange(2), 'col3': np.arange(2)})
    d3 = d1.append(d2, axis=0)
    assert len(d3) == 6


def test_concatenate_df_with_non_overlapping_cols():
    d1 = DDF({'col1': np.arange(4), 'col2': np.arange(4)})
    d2 = DDF({'col1': np.arange(2)})
    d3 = d1.append(d2, axis=0)
    expected_col1 = np.array([0, 1, 2, 3, 0, 1])
    expected_col2 = np.array([0, 1, 2, 3, np.nan, np.nan])
    np.testing.assert_array_equal(d3['col1'], expected_col1)
    np.testing.assert_array_equal(d3['col2'], expected_col2)


def test_appending_not_inplace(df):
    original_df = df.copy(deep=True)
    temp_df = df.rename(lambda col: str(col) + '_suffix')
    df.append(temp_df, axis=1)
    assert df.equals(original_df)
    df.append(temp_df, axis=0)
    assert df.equals(original_df)


def test_cannot_append_wrong_size(df):
    with pytest.raises(ValueError):
        df.append(df[:2], axis=1)


def test_appending_empty_df(df):
    df.append(DDF())

##########################################
# # Slicing
##########################################


def test_boolean_array_slicing(df):
    index = np.random.choice([True, False], len(df))
    sliced = df.rowslice(index)
    assert sliced.equals(df[index])


def test_direct_slicing(df):
    assert df[:5].equals(df.head())
    assert df[-5:].equals(df.tail())
    reverse_ix = np.arange(len(df))[::-1]
    assert df[::-1].equals(df.rowslice(reverse_ix))


def test_cant_slice_with_wrong_size_index(df):
    index = np.repeat(True, 10)
    with pytest.raises(AssertionError):
        df[index]

##########################################
# # Other
##########################################


def test_merge_preservers_strings():
    d1 = DDF({'a': np.arange(4), 'b': np.repeat('b', 4), 'd': np.repeat('r', 4)})
    d2 = DDF({'a': np.arange(4), 'c': np.repeat('c', 4), 'd': np.repeat('2r', 4)})
    merged = d1.merge(d2, on='a')
    assert merged['b'].dtype.type is np.str_
    assert merged['c'].dtype.type is np.str_


def test_can_merge_two_ddfs():
    d1 = DDF({'col1': np.arange(4), 'col2': np.arange(4)})
    d2 = DDF({'col1': np.arange(2), 'col3': np.arange(2)})
    d3 = d2.merge(d1, on='col1', how='outer')
    assert np.allclose(d3['col2'], d1['col2'])
    assert utils.nan_allclose(d3['col3'], np.array([0, 1, np.nan, np.nan]))


def test_set_col(df):
    df['col'] = 0
    assert np.all(df['col'] == np.repeat(0, len(df)))


def test_columns(df, pandas_df):
    df_cols, ddf_cols = pandas_df.columns, df.columns
    assert all(df_col == ddf_col for df_col, ddf_col in zip(df_cols, ddf_cols))


def test_values(df, pandas_df):
    np.testing.assert_allclose(pandas_df.values, df.values)


def test_setitem_scalar_to_single_column(df):
    df['test'] = 123.0
    assert np.all(df['test'] == np.repeat(123.0, len(df)))


def test_setitem_scalar_to_multiple_columns(df):
    df[['test0', 'test1']] = 321.0
    shape = (len(df), 2)
    target_array = np.repeat(321.0, np.prod(shape)).reshape(shape)
    assert np.all(df[['test0', 'test1']] == target_array)


def test_setitem_list_to_single_column(df):
    df['test'] = range(len(df))
    assert np.all(df['test'] == np.arange(len(df)))


def test_setitem_list_to_multiple_column(df):
    list_to_set = [[0, 1]] * len(df)
    df[['test0', 'test1']] = list_to_set
    assert np.all(df[['test0', 'test1']] == np.array(list_to_set))


def test_sort(df):
    df['noise'] = np.random.randn(len(df))
    sorted_noise = np.sort(df['noise'])
    assert not np.allclose(df['noise'], sorted_noise)
    sorted_ddf = df.rowsort('noise')
    assert np.allclose(sorted_ddf['noise'], sorted_noise)
    sorted_ddf = df.rowsort('noise', ascending=False)
    assert np.allclose(sorted_ddf['noise'], sorted_noise[::-1])


def test_equals_returns_false_when_columns_are_not_the_same():
    df = DDF({'col1': np.array([1, 2])})
    df2 = DDF({'col2': np.array([1, 2])})
    assert not df.equals(df2)


def test_equals_invariant_to_column_order():
    df = OrderedDict([
            ('col1', np.array([1, 2])),
            ('col2', np.array([2, 1])),
            ])
    df = DDF(df)
    df2 = OrderedDict([
            ('col2', np.array([2, 1])),
            ('col1', np.array([1, 2])),
            ])
    df2 = DDF(df2)
    assert df.equals(df2)


def test_shallow_copy(df):
    copied = df.copy(deep=False)
    copied['new_col'] = np.arange(len(copied))
    assert 'new_col' not in df


def test_drop_rows(df):
    df['rownr'] = np.arange(len(df))
    rows_to_drop = [1, 5, 10, 15]
    new_df = df.drop_rows(rows_to_drop)
    expected_rownrs = np.array([0, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19])
    assert np.array_equal(new_df['rownr'], expected_rownrs)


def test_colslice_with_filter(df):
    def filter_func(col):
        return str(col).endswith('id')
    new_df = df.colslice(filter_func)
    assert new_df.columns == ['group_id']


def test_renaming_inplace(df):
    df.rename({0: 'renamed_0'}, inplace=True)
    assert 'renamed_0' in df
    assert 0 not in df
    df.rename(lambda col: str(col), inplace=True)
    assert '1' in df
    assert 1 not in df


def test_renaming_not_inplace(df):
    df = df.copy(deep=True)
    new_df = df.rename({0: 'renamed_0'})
    assert 'renamed_0' in new_df
    assert 0 not in new_df
    assert 0 in df
    new_df = df.rename(lambda col: str(col))
    assert '1' in new_df
    assert 1 not in new_df
    assert 1 in df


def test_to_dict(df):
    dct = df.to_dict()
    assert dict(df._data) == dct


def test_colapply(df):
    def replace_with_zero(values):
        return np.zeros(len(values))
    df.colapply(0, replace_with_zero)
    assert np.allclose(df[0], 0)

    original = df.copy(deep=True)
    new_df = df.colapply(0, replace_with_zero, inplace=False)
    assert np.allclose(new_df[0], 0)
    assert np.allclose(original[0], df[0])


def test_drop_duplicates_does_not_change_dtypes(df):
    df['a'] = np.repeat('a', len(df))
    df = df.append(df)
    df = df.drop_duplicates()
    assert df['a'].dtype.type is np.str_


def test_merge_unicode_col():
    bad_df = DDF(DATA)
    bad_str = u'hurac\xe1n'
    bad_df['name'] = np.array([bad_str]*N_EXAMPLES).astype(unicode)
    good_df = DDF(DATA)
    good_str = u'namey_mc_name'
    good_df['name'] = np.array([good_str]*N_EXAMPLES).astype(str)
    bad_df.merge(good_df, on=['group_id', 'name'], how='left')


def test_to_json(df):
    with pytest.raises(AssertionError):
        df.to_json()
    df = df.rename(lambda col: str(col))
    df = df.sort('0')
    json = df.to_json()
    newdf = DDF.from_json(json)
    newdf = newdf.sort('0')
    assert newdf.equals(df)


def test_json_conversion_datetimes():
    dates = np.array([
        '2013-03-30T00:00:00.000000000', '2014-02-10T00:00:00.000000000',
        '2014-05-12T00:00:00.000000000', '2014-03-17T00:00:00.000000000',
        '2013-04-12T00:00:00.000000000', '2014-05-19T00:00:00.000000000',
        '2014-04-09T00:00:00.000000000', '2014-05-11T00:00:00.000000000',
        '2014-02-25T00:00:00.000000000', '2014-04-24T00:00:00.000000000',
        '2014-05-30T00:00:00.000000000', '2014-02-09T00:00:00.000000000',
        '2014-05-05T00:00:00.000000000'], dtype='<M8[ns]')
    df = DDF({'time': dates})
    js = df.to_json()
    newdf = DDF.from_json(js)
    newdf = newdf.colslice(df)
    assert newdf.equals(df)


def test_json_conversion_strings_with_leading_zeros():
    runner_ids = np.array(['000000000' + str(x) for x in range(10)])
    df = DDF({'runner_id': runner_ids})
    js = df.to_json()
    newdf = DDF.from_json(js)
    newdf = newdf.colslice(df)
    assert newdf.equals(df)


def test_json_conversion_preserves_string_dtypes():
    original_df = DDF({
        'runner_id': np.array(['test', 'testing'], dtype='S7'),
        'short_strings': np.array(['a', 'b'], dtype='S1'),
        'more_strings': np.array(['abc', 'def'], dtype='S')
    })

    js = original_df.to_json()
    new_df = DDF.from_json(js)
    new_df = new_df.colslice(original_df)
    assert new_df.equals(original_df)

    for col in ['runner_id', 'short_strings', 'more_strings']:
        assert new_df[col].dtype == original_df[col].dtype
