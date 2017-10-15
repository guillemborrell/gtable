from gtable.fast import union_sorted
import numpy as np
import gtable as gt


def test_gh1_union():
    cal1 = np.array(['2017-04-30', '2017-04-30'], dtype=np.datetime64)
    cal2 = np.array(['2017-05-30', '2017-06-30'], dtype=np.datetime64)

    cal3 = union_sorted(cal1, cal2)

    assert np.all(cal3 == np.array(['2017-04-30', '2017-05-30', '2017-06-30'],
                                   dtype=np.datetime64))


def test_gh1_join():
    t1 = gt.Table(
        {'DATE': [np.datetime64('2017-04-30'), np.datetime64('2017-04-30')],
         'VALUE': [10000000.0, 0.0]}
    )
    t2 = gt.Table(
        {'DATE': [np.datetime64('2017-05-30'), np.datetime64('2017-06-30')]})
    t2.required_columns('VALUE')
    t3 = gt.full_outer_join(t1, t2, 'DATE')
    assert np.all(t3.DATE.values == np.array(
        ['2017-04-30', '2017-04-30',
         '2017-05-30', '2017-06-30'], dtype=np.datetime64))
    assert np.all(t3.VALUE.values == np.array([10000000.0, 0.0]))
    assert np.all(t3.VALUE.index == np.array([1, 1, 0, 0]))
