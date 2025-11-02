import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def example10_2():
    import polars as pl
    from scipy.stats import f

    def i_j(df: pl.DataFrame, column_name: str):
        """Get I and J values from data."""
        i = df.shape[0]
        j = df.get_column(column_name).list.len().min()
        return (i, j)


    def dfn(i):
        """Calculate degrees of freedom for numerator."""
        return i - 1


    def dfd(i, j):
        """Calculate degrees of freedom for denominator."""
        return i * (j - 1)


    def f_critical(alpha, i, j):
        """Calculate f-critical value."""
        ndf = dfn(i)
        ddf = dfd(i, j)
        return f.ppf(1 - alpha, ndf, ddf)


    def mstr(sample_means: pl.Series, i: int, j: int):
        """Calculate mean square for treatments."""
        sum = 0
        grand_mean = sample_means.mean()
        for mean in sample_means:
            sum += pow(mean - grand_mean, 2)
        return (j / (i - 1)) * sum


    def mse(sample_sd: pl.Series, i: int):
        """Calculate mean square error."""
        sum = sample_sd.map_elements(lambda elem: pow(elem, 2)).sum()
        return sum / i


    def f_test(sample_means: pl.Series, sample_sd: pl.Series, i: int, j: int):
        """Perform f-test."""
        _mstr = mstr(sample_means, i, j)
        _mse = mse(sample_sd, i)
        return _mstr / _mse


    def f_test_pvalue(_f_test, i, j):
        """Calculate p-value for f-test."""
        return 1 - f.cdf(_f_test, dfn(i), dfd(i, j))


    # def sst(data: pl.DataFrame):


    def example10_2():
        # Represent data as DataFrame.
        compression_strength_key = "compression_strength_lbs"
        df = (
            pl.DataFrame(
                {
                    "type_of_box": [1, 2, 3, 4],
                    compression_strength_key: [
                        [655.5, 788.3, 734.3, 721.4, 679.1, 699.4],
                        [789.2, 772.5, 786.9, 686.1, 732.1, 774.8],
                        [737.1, 639.0, 696.3, 671.7, 717.2, 727.1],
                        [535.1, 628.7, 542.4, 559.0, 586.9, 520.0],
                    ],
                }
            )
            .with_columns(
                [pl.col(compression_strength_key).list.mean().alias("sample mean")]
            )
            .with_columns(
                [
                    pl.col(compression_strength_key)
                    .list.var()
                    .alias("sample variance")
                ]
            )
            .with_columns(
                [
                    pl.col(compression_strength_key)
                    .list.std()
                    .alias("sample standard deviation")
                ]
            )
        )

        sample_means = df.select("sample mean").to_series()
        sample_sd = df.select("sample standard deviation").to_series()

        alpha = 0.05
        (i, j) = i_j(df, compression_strength_key)
        _f_critical = f_critical(alpha, i, j)
        _f_test = f_test(sample_means, sample_sd, i, j)

        print(f"f-critical value: {_f_critical}")
        print(f"f-test: {_f_test}")


    def example10_3():
        _sample_means = pl.Series([10.5, 14.8, 15.7, 16.0, 21.6])
        _sample_sd = pl.Series([4.5, 6.8, 6.5, 6.7, 6.0])
        _i = 5
        _j = 10

        alpha = 0.01
        _f_critical = f_critical(alpha, _i, _j)
        _f_test = f_test(_sample_means, _sample_sd, _i, _j)
        _p_value = f_test_pvalue(_f_test, _i, _j)

        print(f"f-critical value: {_f_critical}")
        print(f"f-test: {_f_test}")
        print(f"p-value: {_p_value}")


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
        _f_test = f_test(_sample_means, _sample_sd, _i, _j)
        _p_value = f_test_pvalue(_f_test, _i, _j)
        _mstr = mstr(_sample_means, _i, _j)
        _mse = mse(_sample_sd, _i)

        print(f"f-critical value: {_f_critical}")
        print(f"mstr: {_mstr}")
        print(f"mse: {_mse}")
        print(f"f-test: {_f_test}")
        print(f"p-value: {_p_value}")


    def problem6():
        df = (
            pl.DataFrame(
                {
                    "data": [
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
                }
            )
            .with_columns([pl.col("data").list.mean().alias("sample mean")])
            .with_columns([pl.col("data").list.var().alias("sample variance")])
            .with_columns(
                [pl.col("data").list.std().alias("sample standard deviation")]
            )
        )

        sample_means = df.select("sample mean").to_series()
        sample_sd = df.select("sample standard deviation").to_series()

        alpha = 0.01
        (i, j) = i_j(df, "data")
        _f_critical = f_critical(alpha, i, j)
        _f_test = f_test(sample_means, sample_sd, i, j)

        print(f"f-critical value: {_f_critical}")
        print(f"f-test: {_f_test}")


    problem6()
    return


if __name__ == "__main__":
    app.run()
