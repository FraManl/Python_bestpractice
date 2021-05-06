import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
from matplotlib import pyplot as plt


def _pdf_cdf(x):
    return pd.Series([norm.pdf(x), norm.cdf(x)])


def _compute(dataset, col):
    """
    Compute PCF and PDF for a given dataset column
    :param dataset:
    :param col:
    :return:
    """

    dataset[['pdf', 'cdf']] = dataset[col].apply(_pdf_cdf)
    return dataset


start = -7
end = 7
step = 0.001
df = pd.DataFrame(
    {
        'col1': np.arange(start, end, step),
        'col2': np.arange(start, end, step),
        'col3': np.arange(start, end, step),
    }
)

print(df.head())

_col1_data = _compute(df, 'col1')
_col2_data = _compute(df, 'col2')
