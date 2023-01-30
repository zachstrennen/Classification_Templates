import pandas as pd
from

for df in pd.read_csv(path, chunksize=10_000):

    #find a dataset and link the name of the dataset
    #why that dataset?
    #add a docstring to repo for longrun

    #add dataset to github

def read_data(path: str) -> pd.DataFrame:
    """Read in CMS data of payments
    Args:
    path (str): location on disk

    returns:
    pd.DataFrame: a dataframe of provider payments

    :param path:
    :return:
    """

    def cal_vs_wyoming(df:pd.DataFrame) -> tuple[float,float]:
        """
        How much money in CA vs
        :param df:
        :return:
        """
        ca = df.loc[df['Recipient_State']=='CA','Total_Amount_Invested_US'].sum()
        wy = df.loc[df['Recipient_State'] == 'WY', 'Total_Amount_Invested_US'].sum()
        return ca, wy

if __name__ == '__main__':
    a = 1
    df = read_data('data/PATHHERE')
    print(df)