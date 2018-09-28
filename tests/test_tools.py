import numpy as np

from ddf import DDF, concatenate, append_csv, thin_by_id


def test_concatenate():
    data = {'a': np.repeat('A', 10), 'b': np.repeat(10, 10)}
    df = DDF(data)
    new_df = concatenate([df]*100)
    assert new_df.shape == (1000, 2)
    assert new_df['a'].dtype == df['a'].dtype
    assert new_df['b'].dtype == df['b'].dtype


def test_append_csv(tmpdir):
    path = str(tmpdir) + 'log.csv'
    data = DDF({'a': 1, 'b': 10})
    append_csv(data, path)
    loaded = DDF.from_csv(path)
    assert loaded.equals(data)

    new_data = DDF({'a': 100, 'b': 101})
    append_csv(new_data, path)
    loaded_appended = DDF.from_csv(path)
    appended = data.append(new_data)
    assert appended.equals(loaded_appended)


def test_thin_by_id_retains_df_order():
    df = DDF({'race_id': [5, 5, 5, 6, 6, 3, 3, 3, 4, 4, 1, 1]})
    result = thin_by_id(df, 'race_id', 2)
    expected = np.array([5, 5, 5, 3, 3, 3, 1, 1])
    assert np.all(result['race_id'] == expected)


def test_thin_by_id_with_offset():
    df = DDF({'race_id': [5, 5, 5, 6, 6, 3, 3, 3, 4, 4, 1, 1]})
    result = thin_by_id(df, 'race_id', 2, offset=1)
    expected = np.array([6, 6, 4, 4])
    assert np.all(result['race_id'] == expected)
