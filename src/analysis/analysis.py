import pickle
import re
import statistics

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from bld.project_paths import project_paths_join as ppj
from src.data_management.clean_data import identify_quarter


def sentiment_fraction(quarter):
    """fraction of tweets with sentiment_coeff > 0
    """
    counter = 0
    for val in quarter:

        if val > 0:
            counter += 1
    return counter / len(quarter)


def load_historical_data(file_name, date_column, var_name):
    """Loads and formats historical data to match the quarters_sentiment constructed previously.
    1 row for every quarter, columns are variables of interest

    Inputs:
    file_name: name of the file in IN_DATA
    date_column: name of the column with the date_list
    var_name: list with the names of the variables of interest
    """
    df = pd.read_csv(ppj("IN_DATA", file_name))
    df[date_column] = df[date_column].apply(lambda x: re.sub(r"-", "", x))
    df[date_column] = df[date_column].apply(lambda x: x[:4]) + df[date_column].apply(
        lambda x: identify_quarter(x)
    )
    df = df.set_index(date_column)
    df.columns = var_name
    return df


def regression_tables(macro_ind):
    """creates a regression table as latex for each variable in macro_ind
        macro_ind: list of name of variabels of interest as strings
    """
    for ind in macro_ind:

        mod = smf.ols(formula=ind + "~ fraction ", data=df)
        res1 = mod.fit()

        mod = smf.ols(formula=ind + "~ mean ", data=df)
        res2 = mod.fit()

        mod = smf.ols(formula=ind + "~ var ", data=df)
        res3 = mod.fit()

        mod = smf.ols(formula=ind + "~ var + mean + var ", data=df)
        res4 = mod.fit()

        textfile = open(ppj("OUT_ANALYSIS", ind + "_on_sentiment.txt"), "w")
        textfile.write(
            summary_col(
                [res1, res2, res3, res4],
                stars=True,
                float_format="%0.2f",
                model_names=["\n(0)", "\n(1)", "\n(2)", "\n(3)"],
                info_dict={
                    "N": lambda x: "{:d}".format(int(x.nobs)),
                    "R2": lambda x: f"{x.rsquared:.2f}",
                },
            ).as_latex()
        )
        textfile.close()


def regression_plots(macro_ind, indep_var):
    """creates regression plots for each dependent variable on each dependent variable.
    """
    for y in macro_ind:
        for x in indep_var:
            ax1 = sns.regplot(y=y, x=x, data=df)
            ax1.set(xlabel=x, ylabel=y)
            ax1.get_figure().savefig(ppj("OUT_FIGURES", y + "_on_" + x))


if __name__ == "__main__":

    data = pickle.load(open(ppj("OUT_DATA", "data_weighted_coeff.pickle"), "rb"))

    quarter_values = dict.fromkeys(data.keys(), [])
    quarter_values
    for quarter in data:
        quarter_values[quarter] = {}
        quarter_values[quarter]["mean"] = statistics.mean(data[quarter])
        quarter_values[quarter]["var"] = statistics.variance(data[quarter])
        quarter_values[quarter]["fraction"] = sentiment_fraction(data[quarter])

    # load macroeconomic indicators
    # GDP
    gdp = load_historical_data("real_gdp_US.csv", "DATE", ["GDP"])
    # Unemployment
    unemploy = load_historical_data(
        "unemployment_rate_US.csv", "DATE", ["UNEMPLOYMENT"]
    )
    # exchange rate USD EURO
    exchange = load_historical_data("USDEURO_exchange_rate.csv", "DATE", ["USDEURO"])
    # CPI
    cpi = load_historical_data("consumer_price_index_US.csv", "DATE", ["CPI"])

    macro_ind = ["GDP", "UNEMPLOYMENT", "USDEURO", "CPI"]

    # Concatenate dataframe with variables
    df = pd.DataFrame.from_dict(quarter_values).T
    indep_var = df.columns
    df = pd.concat([df, gdp, unemploy, exchange, cpi], axis=1)
    df.to_excel(ppj("OUT_DATA", "sentiment_and_interest_var.xlsx"))

    # OLS
    regression_tables(macro_ind)

    # simple Regression Plots
    sns.set(color_codes=True)
    regression_plots(macro_ind, indep_var)
