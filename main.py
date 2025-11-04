import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def chapter10():
    import math
    import polars as pl
    import numpy as np
    from scipy.stats import f
    from typing import List
    from scipy.stats import studentized_range


    def total_df(i, j):
        """Get total degrees of freedom."""
        return i * j - 1


    def dfn(i):
        """Calculate degrees of freedom for numerator (treatments)."""
        return i - 1


    def dfd(i, j):
        """Calculate degrees of freedom for denominator (error)."""
        return i * (j - 1)


    def f_critical(alpha, i, j):
        """Calculate f-critical value."""
        ndf = dfn(i)
        ddf = dfd(i, j)
        return f.ppf(1 - alpha, ndf, ddf)


    def grand_mean(sample_means: pl.Series):
        """Calculate the grand mean."""
        return sample_means.mean()


    def mstr(sample_means: pl.Series, i: int, j: int):
        """Calculate mean square for treatments."""
        _sum = 0
        _grand_mean = grand_mean(sample_means)
        for _mean in sample_means:
            _sum += pow(_mean - _grand_mean, 2)
        return (j / (i - 1)) * _sum


    def mse(sample_sd: pl.Series, i: int):
        """Calculate mean square error."""
        sum = sample_sd.map_elements(lambda elem: pow(elem, 2)).sum()
        return sum / i


    def f_test_statistic(sample_means: pl.Series, sample_sd: pl.Series, i: int, j: int):
        """Get f test statistic."""
        _mstr = mstr(sample_means, i, j)
        _mse = mse(sample_sd, i)
        return _mstr / _mse


    def f_test_pvalue(_f_test_statistic, i, j):
        """Calculate p-value for f-test."""
        return 1 - f.cdf(_f_test_statistic, dfn(i), dfd(i, j))

    def q_critical(_alpha, _i, _j):
        return studentized_range.ppf(1-_alpha, _i, dfd(_i, _j))

    def w(_sample_sd: pl.Series, _i: int, _j: int, _alpha: float):
        _q_critical = q_critical(_alpha, _i, _j)
        _sqrt_mse_j = math.sqrt(mse(_sample_sd, _i)/_j)
        return _q_critical * _sqrt_mse_j

    def sorted_sample_means(_sample_means: pl.Series):
        return _sample_means.sort()


    class AnovaWithData:
        data_key = "data"
        s_mean_key = "sample mean"
        s_var_key = "sample variance"
        s_sd_key = "sample standard deviation"

        def __init__(self, data: List[List[float]], alpha: float = 0.05):
            self.alpha = alpha
            self.df = (
                pl.DataFrame(
                    {
                        AnovaWithData.data_key: data,
                    }
                )
                .with_columns(
                    [
                        pl.col(AnovaWithData.data_key)
                        .list.mean()
                        .alias(AnovaWithData.s_mean_key)
                    ]
                )
                .with_columns(
                    [
                        pl.col(AnovaWithData.data_key)
                        .list.var()
                        .alias(AnovaWithData.s_var_key)
                    ]
                )
                .with_columns(
                    [
                        pl.col(AnovaWithData.data_key)
                        .list.std()
                        .alias(AnovaWithData.s_sd_key)
                    ]
                )
            )

        def sample_means(self):
            """Get sample means."""
            return self.df.select(AnovaWithData.s_mean_key).to_series()

        def sample_sd(self):
            """Get sample standard deviations."""
            return self.df.select(AnovaWithData.s_sd_key).to_series()

        def sorted_sample_means(self):
            return sorted_sample_means(self.sample_means())

        def grand_mean(self):
            """Calculate grand mean."""
            return grand_mean(self.sample_means())

        def grand_sum(self):
            """Calculate grand sum."""
            return np.concatenate(
                self.df[AnovaWithData.data_key].to_numpy()
            ).sum()

        def i(self):
            """Get the i value."""
            return self.df.shape[0]

        def j(self):
            """Get the j value."""
            len_list = self.df.get_column(
                AnovaWithData.data_key
            ).list.len()
            assert len_list.n_unique() == 1
            return len_list.first()

        def total_df(self):
            """Get total degrees of freedom."""
            return total_df(self.i(), self.j())

        def dfn(self):
            """Calculate degrees of freedom for numerator (treatments)."""
            return dfn(self.i())

        def dfd(self):
            """Calculate degrees of freedom for denominator."""
            return dfd(self.i(), self.j())

        def f_critical(self):
            """Calculate f-critical value."""
            return f_critical(self.alpha, self.i(), self.j())

        def mstr(self):
            """Calculate mean square for treatments."""
            sample_means = self.sample_means()
            return mstr(sample_means, self.i(), self.j())

        def mse(self):
            """Calculate mean square error."""
            sample_sd = self.sample_sd()
            return mse(sample_sd, self.i())

        def sst(self):
            """Calculate the total sum squares."""
            _ij = self.i() * self.j()
            _flattened = np.concatenate(
                self.df[AnovaWithData.data_key].to_numpy()
            )
            _first_term = np.square(_flattened).sum()
            return _first_term - (pow(self.grand_sum(), 2) / _ij)

        def sstr(self):
            """Calculate the treatment sum squares."""
            _rows = self.df[AnovaWithData.data_key].to_numpy()
            _first_term = (
                np.array([_row.sum() ** 2 for _row in _rows]).sum() / self.j()
            )
            _second_term = pow(self.grand_sum(), 2) / (self.i() * self.j())
            return _first_term - _second_term

        def sse(self):
            """Calculate error of sum squares."""
            _df = (
                pl.DataFrame({"data": self.df[AnovaWithData.data_key]})
                .with_columns([pl.col("data").list.mean().alias("mean")])
                .with_columns(
                    [
                        pl.struct(["data", "mean"])
                        .map_elements(
                            lambda x: [pow(v - x["mean"], 2) for v in x["data"]],
                            return_dtype=pl.List(pl.Float64),
                        )
                        .alias("data_minus_mean")
                    ]
                )
            )
            return np.concatenate(_df["data_minus_mean"].to_numpy()).sum()

        def f_test_statistic(self):
            """Get f test statistic."""
            sample_means = self.sample_means()
            sample_sd = self.sample_sd()
            return f_test_statistic(sample_means, sample_sd, self.i(), self.j())

        def f_test_pvalue(self):
            """Calculate p-value for f-test."""
            _f_test = self.f_test_statistic()
            return 1 - f.cdf(_f_test, self.dfn(), self.dfd())

        def w(self):
            return w(self.sample_sd(), self.i(), self.j(), self.alpha)

        def q_critical(self):
            return q_critical(self.alpha, self.i(), self.j())

        def print(self):
            print(f"treatments degrees of freedom: {self.dfn()}")
            print(f"treatment sum of squares (SSTr): {self.sstr()}")
            print(f"mean square for treatments (MSTr): {self.mstr()}")
            print("---------------------------------------------------")
            print(f"error degrees of freedom: {self.dfd()}")
            print(f"error sum of squares (SSE): {self.sse()}")
            print(f"mean square for error (MSE): {self.mse()}")
            print("---------------------------------------------------")
            print(f"total degrees of freedom: {self.total_df()}")
            print(f"total sum of squares (SST): {self.sst()}")
            print("---------------------------------------------------")
            print(f"f critical value: {self.f_critical()}")
            print(f"f test statistic: {self.f_test_statistic()}")
            print(f"f test p value: {self.f_test_pvalue()}")


    class AnovaWithoutData:
        def __init__(self, sample_means: pl.Series, sample_sd: pl.Series, i: int, j: int, alpha: float = 0.05):
            self.sample_means = sample_means
            self.sample_sd = sample_sd
            self.i = i
            self.j = j
            self.alpha = alpha

        def sorted_sample_means(self):
            return sorted_sample_means(self.sample_means)

        def grand_mean(self):
            """Calculate the grand mean."""
            return grand_mean(self.sample_means)

        def dfn(self):
            """Calculate degrees of freedom for numerator (treatments)."""
            return dfn(self.i)

        def dfd(self):
            """Calculate degrees of freedom for denominator (error)."""
            return dfd(self.i, self.j)

        def total_df(self):
            """Get total degrees of freedom."""
            return total_df(self.i, self.j)

        def mstr(self):
            """Calculate mean square for treatments."""
            return mstr(self.sample_means, self.i, self.j)

        def mse(self):
            """Calculate mean square error."""
            return mse(self.sample_sd, self.i)

        def f_test_statistic(self):
            """Get f test statistic."""
            return f_test_statistic(self.sample_means, self.sample_sd, self.i, self.j)

        def f_test_pvalue(self):
            """Calculate p-value for f-test."""
            return f_test_pvalue(self.f_test_statistic(), self.i, self.j)

        def f_critical(self):
            """Calculate f-critical value."""
            return f_critical(self.alpha, self.i, self.j)

        def sst(self):
            """Calculate the total sum squares."""
            return self.sstr() + self.sse()

        def sstr(self):
            """Calculate the treatment sum squares."""
            return self.mstr() * self.dfn()

        def sse(self):
            """Calculate error of sum squares."""
            return self.mse() * self.dfd()

        def w(self):
            return w(self.sample_sd, self.i, self.j, self.alpha)

        def q_critical(self):
            return q_critical(self.alpha, self.i, self.j)

        def print(self):
            print(f"treatments degrees of freedom: {self.dfn()}")
            print(f"treatment sum of squares (SSTr): {self.sstr()}")
            print(f"mean square for treatments (MSTr): {self.mstr()}")
            print("---------------------------------------------------")
            print(f"error degrees of freedom: {self.dfd()}")
            print(f"error sum of squares (SSE): {self.sse()}")
            print(f"mean square for error (MSE): {self.mse()}")
            print("---------------------------------------------------")
            print(f"total degrees of freedom: {self.total_df()}")
            print(f"total sum of squares (SST): {self.sst()}")
            print("---------------------------------------------------")
            print(f"f critical value: {self.f_critical()}")
            print(f"f test statistic: {self.f_test_statistic()}")
            print(f"f test p value: {self.f_test_pvalue()}")


    def section10_1():
        def example10_2():
            data = [
                [655.5, 788.3, 734.3, 721.4, 679.1, 699.4],
                [789.2, 772.5, 786.9, 686.1, 732.1, 774.8],
                [737.1, 639.0, 696.3, 671.7, 717.2, 727.1],
                [535.1, 628.7, 542.4, 559.0, 586.9, 520.0],
            ]
            data = AnovaWithData(data)
            data.print()

        def example10_3():
            _sample_means = pl.Series([10.5, 14.8, 15.7, 16.0, 21.6])
            _sample_sd = pl.Series([4.5, 6.8, 6.5, 6.7, 6.0])
            _i = 5
            _j = 10
            _alpha = 0.01
            _anova = AnovaWithoutData(_sample_means, _sample_sd, _i, _j, _alpha)
            _anova.print()

        def example10_4():
            data = [
                [0.56, 1.12, 0.90, 1.07, 0.94],
                [0.72, 0.69, 0.87, 0.78, 0.91],
                [0.62, 1.08, 1.07, 0.99, 0.93],
            ]
            data = AnovaWithData(data, alpha=0.01)
            data.print()

        def problem1():
            """Page 400"""
            _i = 5
            _j = 4
            _mstr = 2673.3
            _mse = 1094.2

            def a():
                _alpha = 0.05
                _f_critical = f_critical(_alpha, _i, _j)
                _f_test = _mstr / _mse
                print(f"f-critical: {_f_critical}")
                print(f"f-test: {_f_test}")

            def b():
                pvalue = f_test_pvalue(_mstr / _mse, _i, _j)
                print(f"pvalue: {pvalue}")

            a()
            b()

        def problem5():
            _sample_means = pl.Series([1.63, 1.56, 1.42])
            _sample_sd = pl.Series([0.27, 0.24, 0.26])
            _i = 3
            _j = 10
            _alpha = 0.05

            _f_critical = f_critical(_alpha, _i, _j)
            _f_test = f_test_statistic(_sample_means, _sample_sd, _i, _j)
            _p_value = f_test_pvalue(_f_test, _i, _j)
            _mstr = mstr(_sample_means, _i, _j)
            _mse = mse(_sample_sd, _i)

            print(f"f-critical value: {_f_critical}")
            print(f"mstr: {_mstr}")
            print(f"mse: {_mse}")
            print(f"f-test: {_f_test}")
            print(f"p-value: {_p_value}")

        def problem6():
            data = [
                [
                    20.5,
                    28.1,
                    27.8,
                    27.0,
                    28.0,
                    25.2,
                    25.3,
                    27.1,
                    20.5,
                    31.3,
                ],
                [
                    26.3,
                    24.0,
                    26.2,
                    20.2,
                    23.7,
                    34.0,
                    17.1,
                    26.8,
                    23.7,
                    24.9,
                ],
                [
                    29.5,
                    34.0,
                    27.5,
                    29.4,
                    27.9,
                    26.2,
                    29.9,
                    29.5,
                    30.0,
                    35.6,
                ],
                [
                    36.5,
                    44.2,
                    34.1,
                    30.3,
                    31.4,
                    33.1,
                    34.1,
                    32.9,
                    36.3,
                    25.5,
                ],
            ]
            _anova = AnovaWithData(data)
            _anova.print()

        def problem8():
            _data = [
                [309.2, 409.5, 311.0, 326.5, 316.8, 349.8, 309.7],
                [402.1, 347.2, 361.0, 404.5, 331.0, 348.9, 381.7],
                [392.4, 366.2, 351.0, 357.1, 409.9, 367.3, 382.0],
                [346.7, 452.9, 461.4, 433.1, 410.6, 384.2, 362.6],
                [407.4, 441.8, 419.9, 410.7, 473.4, 441.2, 465.8],
            ]
            _anova = AnovaWithData(_data)
            _anova.print()

        example10_3()

    def section10_2():
        def example10_6():
            _data = [
                [88.6, 73.2, 91.4, 68.0, 75.2],
                [63.0, 53.9, 69.2, 50.1, 71.5],
                [44.9, 59.5, 40.2, 56.3, 38.7],
                [31.0, 39.6, 45.3, 25.2, 22.7],
            ]
            _alpha = 0.05
            _anova = AnovaWithData(_data, _alpha)
            _anova.print()
            print(f"w: {_anova.w()}")
            print(f"q critical: {_anova.q_critical()}")
            print(f"sorted sample means: {_anova.sorted_sample_means()}")


        def problem14():
            _sample_means = pl.Series([10.5, 14.8, 15.7, 16.0, 21.6])
            _sample_sd = pl.Series([4.5, 6.8, 6.5, 6.7, 6.0])
            _i = 5
            _j = 10
            _alpha = 0.01
            _anova = AnovaWithoutData(_sample_means, _sample_sd, _i, _j, _alpha)
            _anova.print()
            print("----------------------------------------------------")
            print(f"w: {_anova.w()}")
            print(f"q critical: {_anova.q_critical()}")
            print(f"sorted sample means: {_anova.sorted_sample_means()}")

        def problem18():
            _data = [
                [13, 17, 7, 14],
                [21, 13, 20, 17],
                [18, 15, 20, 17],
                [7, 11, 18, 10],
                [6, 11, 15, 8],
            ]
            _anova = AnovaWithData(_data, 0.05)
            _anova.print()
            print("----------------------------------------------------")
            print(f"w: {_anova.w()}")
            print(f"q critical: {_anova.q_critical()}")
            print(f"sorted sample means: {_anova.sorted_sample_means()}")

        problem18()

    section10_2()
    return


if __name__ == "__main__":
    app.run()
