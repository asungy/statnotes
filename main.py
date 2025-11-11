import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def two_factor_anova():
    import polars as pl
    from scipy.stats import f
    from typing import List

    class TwoFactorAnova:
        def __init__(
            self,
            alpha: float = 0.05,
            data: List[List[float]] = None,
            params: dict[str, any] = {},
        ):
            self._alpha = alpha
            if data:
                self._data = pl.DataFrame(data).select(
                    pl.all().map_elements(
                        lambda x: (
                            [x] if isinstance(x, float) or isinstance(x, int) else x
                        )
                    )
                )
            #####################################
            # I and J and K
            #####################################
            self._i: int = (
                (data and self.__class__.i(self._data))
                or params.get("i")
                or print("no value assigned for i")
            )
            self._j: int = (
                (data and self.__class__.j(self._data))
                or params.get("j")
                or print("no value assigned for j")
            )
            self._k: int = (
                (data and self.__class__.k(self._data))
                or params.get("k")
                or print("no value assigned for k")
            )
            #####################################
            # Degrees of freedom
            #####################################
            # Factor A df
            self._adf: int = self.adf()
            # Factor B df
            self._bdf: int = self.bdf()
            # Error df
            self._edf: int = self.edf()
            # Total df
            self._tdf: int = self.tdf()
            # Interaction df
            self._abdf: int = self.abdf()
            #####################################
            # Sum of Squares
            #####################################
            # Factor A sum of squares
            self._ssa: float = (
                (data and self.__class__.data_to_ssa(self._data))
                or params.get("ssa")
                or print("no value assigned for ssa")
            )
            # Factor B sum of squares
            self._ssb: float = (
                (data and self.__class__.data_to_ssb(self._data))
                or params.get("ssb")
                or print("no value assigned for ssb")
            )
            # Error sum of squares
            self._sse: float = (
                (data and self.__class__.data_to_sse(self._data))
                or params.get("sse")
                or print("no value assigned for sse")
            )
            # Interaction sum of squares.
            self._ssab: float = (
                (data and self.__class__.data_to_ssab(self._data))
                or params.get("ssab")
                or print("no value assigned for ssab")
            )
            # Total sum of squares
            self._sst: float = (
                (data and self.__class__.data_to_sst(self._data))
                or params.get("sst")
                or (self._ssa + self._ssb + self._sse + self._ssab)
            )
            #####################################
            # Mean Squares
            #####################################
            # Factor A mean square
            self._msa: float = self.msa()
            # Factor B mean square
            self._msb: float = self.msb()
            # Error mean square
            self._mse: float = self.mse()
            # Interaction mean square
            self._msab: float = self.msab()
            #####################################
            # F test statistic
            #####################################
            # f test statistic of factor A
            self._f_a: float = self.f_a()
            # f_a critical value
            self._f_a_critical_value: float = (
                lambda alpha: f.ppf(1 - alpha, self.adf(), self.edf())
            )(alpha)
            # f_a pvalue
            self._f_a_pvalue: float = (lambda f_a, adf, edf: 1 - f.cdf(f_a, adf, edf))(
                self.f_a(), self.adf(), self.edf()
            )
            # f test statistic of factor B
            self._f_b: float = self.f_b()
            # f_b critical value
            self._f_b_critical_value: float = (
                lambda alpha: f.ppf(1 - alpha, self.bdf(), self.edf())
            )(alpha)
            # f_b pvalue
            self._f_b_pvalue: float = (lambda f_b, bdf, edf: 1 - f.cdf(f_b, bdf, edf))(
                self.f_b(), self.bdf(), self.edf()
            )
            # f test statistic of interaction
            self._f_ab: float = self.f_ab()
            # f_ab critical value
            self._f_ab_critical_value: float = (
                lambda alpha: f.ppf(1 - alpha, self.abdf(), self.edf())
            )(alpha)
            # f_ab pvalue
            self._f_ab_pvalue: float = (
                lambda f_ab, abdf, edf: 1 - f.cdf(f_ab, abdf, edf)
            )(self.f_ab(), self.abdf(), self.edf())

        @classmethod
        def from_data(cls, data: List[List[float]], alpha: float):
            assert data is not None
            assert alpha is not None
            return cls(
                alpha,
                data,
                {},
            )

        @classmethod
        def from_params(cls, params: dict[str, any], alpha: float):
            assert params is not None
            assert alpha is not None
            return cls(
                alpha,
                None,
                params,
            )

        def print(self):
            print("-----------------------------------")
            print(f"i: {self._i}")
            print(f"j: {self._j}")
            print(f"k: {self._k}")
            print("-----------------------------------")
            print(f"Factor A df: {self._adf}")
            print(f"Factor B df: {self._bdf}")
            print(f"Interaction df: {self._abdf}")
            print(f"Error df: {self._edf}")
            print(f"Total df: {self._tdf}")
            print("-----------------------------------")
            print(f"Factor A sum of squares: {self._ssa}")
            print(f"Factor B sum of squares: {self._ssb}")
            print(f"Interaction sum of squares: {self._ssab}")
            print(f"Error sum of squares: {self._sse}")
            print(f"Total sum of squares: {self._sst}")
            print("-----------------------------------")
            print(f"Factor A mean square: {self._msa}")
            print(f"Factor B mean square: {self._msb}")
            print(f"Interaction mean square: {self._msab}")
            print(f"Error mean square: {self._mse}")
            print("-----------------------------------")
            print(f"alpha: {self._alpha}")
            print(f"f test statistic of factor A: {self._f_a}")
            print(f"f_a critical value: {self._f_a_critical_value}")
            print(f"f_a p-value: {self._f_a_pvalue}")
            print()
            print(f"f test statistic of factor B: {self._f_b}")
            print(f"f_b critical value: {self._f_b_critical_value}")
            print(f"f_b p-value: {self._f_b_pvalue}")
            print()
            print(f"f test statistic of interaction: {self._f_ab}")
            print(f"f_ab critical value: {self._f_ab_critical_value}")
            print(f"f_ab p-value: {self._f_ab_pvalue}")
            print("-----------------------------------")

        @classmethod
        def i(cls, data: pl.DataFrame):
            return data.shape[1]

        @classmethod
        def j(cls, data: pl.DataFrame):
            return data.shape[0]

        @classmethod
        def k(cls, data: pl.DataFrame):
            return (
                data.select(pl.all().map_elements(lambda x: x.len())).to_numpy().min()
            )

        def adf(self):
            return self._i - 1

        def bdf(self):
            return self._j - 1

        def edf(self):
            if self._k == 1:
                return self._adf * self._bdf
            return self._i * self._j * (self._k - 1)

        def abdf(self):
            return (self._i - 1) * (self._j - 1)

        @classmethod
        def data_to_tdf(cls, data: pl.DataFrame):
            return cls.i(data) * cls.j(data) - 1

        def tdf(self):
            return self._i * self._j * self._k - 1

        @classmethod
        def grand_mean(cls, data: pl.DataFrame):
            return data.to_numpy().mean().mean()

        @classmethod
        def data_to_ssa(cls, data: pl.DataFrame):
            grand_mean = cls.grand_mean(data)
            j = cls.j(data)
            k = cls.k(data)
            return (
                (
                    data.select(pl.all().map_elements(lambda x: x.mean()))
                    .mean()
                    .select(pl.all().map_elements(lambda x: pow(x - grand_mean, 2)))
                    .to_numpy()
                    .sum()
                )
                * j
                * k
            )

        @classmethod
        def data_to_ssb(cls, data: pl.DataFrame):
            grand_mean = cls.grand_mean(data)
            i = cls.i(data)
            k = cls.k(data)
            return (
                (
                    data.transpose()
                    .select(pl.all().map_elements(lambda x: x.mean()))
                    .mean()
                    .select(pl.all().map_elements(lambda x: pow(x - grand_mean, 2)))
                    .to_numpy()
                    .sum()
                )
                * i
                * k
            )

        @classmethod
        def data_to_sse(cls, data: pl.DataFrame):
            k = cls.k(data)
            if k == 1:
                result = 0
                flat_data = data.select(pl.all().map_elements(lambda x: x[0]))
                x_i = flat_data.mean().to_numpy()[0]
                x_j = flat_data.transpose().mean().to_numpy()[0]
                nd = flat_data.transpose().to_numpy()
                grand_mean = TwoFactorAnova.grand_mean(data)
                for i in range(nd.shape[0]):
                    for j in range(nd.shape[1]):
                        result += pow(nd[i][j] - x_i[i] - x_j[j] + grand_mean, 2)
                return result

            return (
                (data - data.select(pl.all().map_elements(lambda x: [x.mean()] * k)))
                .select(
                    pl.all().map_elements(
                        lambda x: x.map_elements(lambda s: pow(s, 2)).sum()
                    )
                )
                .to_numpy()
                .sum()
            )

        @classmethod
        def data_to_sst(cls, data: pl.DataFrame):
            grand_mean = TwoFactorAnova.grand_mean(data)
            return (
                data.select(
                    pl.all().map_elements(
                        lambda x: x.map_elements(lambda s: pow(s - grand_mean, 2)).sum()
                    )
                )
                .to_numpy()
                .sum()
            )

        @classmethod
        def data_to_ssab(cls, data: pl.DataFrame):
            i = cls.i(data)
            j = cls.j(data)
            k = cls.k(data)
            grand_mean = cls.grand_mean(data)

            x_ij_mean = data.select(pl.all().map_elements(lambda x: x.mean()))

            x_i_mean = pl.concat(
                [data.select(pl.all().map_elements(lambda x: x.mean())).mean()] * j
            )

            x_j_mean = pl.concat(
                [
                    data.transpose()
                    .select(pl.all().map_elements(lambda x: x.mean()))
                    .mean()
                ]
                * i
            ).transpose()

            return (x_ij_mean - x_i_mean - x_j_mean + grand_mean).select(
                pl.all().map_elements(lambda x: pow(x, 2))
            ).to_numpy().sum() * k

        def msa(self):
            return self._ssa / self._adf

        def msb(self):
            return self._ssb / self._bdf

        def mse(self):
            return self._sse / self._edf

        def msab(self):
            return self._ssab / self._abdf

        def f_a(self):
            return self._msa / self._mse

        def f_b(self):
            return self._msb / self._mse

        def f_ab(self):
            return self._msab / self._mse

        @classmethod
        def fixed_effects_alpha(cls, data: pl.DataFrame, i):
            return data.transpose().to_numpy()[i - 1].mean().mean() - cls.grand_mean(data)

        @classmethod
        def fixed_effects_beta(cls, data: pl.DataFrame, j):
            return data.to_numpy()[j - 1].mean().mean() - cls.grand_mean(data)
    return (TwoFactorAnova,)


@app.cell
def example11_1_from_data(TwoFactorAnova):
    def _():
        data = [
            [0.97, 0.48, 0.48, 0.46],
            [0.77, 0.14, 0.22, 0.25],
            [0.67, 0.39, 0.57, 0.19],
        ]
        anova = TwoFactorAnova.from_data(data, 0.05)
        anova.print()

    _()
    return


@app.cell(disabled=True)
def example11_1_from_params(TwoFactorAnova):
    def _():
        params = {
            "i": 3,
            "j": 4,
            "ssa": 0.12821666666666667,
            "ssb": 0.47969166666666674,
            "sse": 0.08678333333333331,
        }
        anova = TwoFactorAnova.from_params(params, 0.05)
        anova.print()

    _()
    return


@app.cell
def fixed_effects_example(TwoFactorAnova):
    def _():
        import polars as pl

        data = pl.DataFrame(
            [
                [2, 5],
                [3, 6],
            ]
        )
        print(TwoFactorAnova.fixed_effects_alpha(data, 1))
        print(TwoFactorAnova.fixed_effects_alpha(data, 2))
        print(TwoFactorAnova.fixed_effects_beta(data, 1))
        print(TwoFactorAnova.fixed_effects_beta(data, 2))

    _()
    return


@app.cell
def question2(TwoFactorAnova):
    def _():
        import polars as pl

        data = [
            [64, 49, 50],
            [53, 51, 48],
            [47, 45, 50],
            [51, 43, 52],
        ]
        anova = TwoFactorAnova.from_data(data, 0.05)
        anova.print()
        print(f"ssa - ssb: {anova._ssa - anova._ssb}")

        df = pl.DataFrame(data)
        beta_hat_1 = TwoFactorAnova.fixed_effects_beta(df, 1)
        beta_hat_3 = TwoFactorAnova.fixed_effects_beta(df, 3)
        print(f"beta_hat_1 - beta_hat_3: {beta_hat_1 - beta_hat_3}")
        alpha_hat_1 = TwoFactorAnova.fixed_effects_alpha(df, 1)
        alpha_hat_4 = TwoFactorAnova.fixed_effects_alpha(df, 4)
        print(f"alpha_hat_1 - alpha_hat_4: {alpha_hat_1 - alpha_hat_4}")
        grand_mean = TwoFactorAnova.grand_mean(df)
        print(f"mu_hat (grand mean): {grand_mean}")

    _()
    return


@app.cell(disabled=True)
def question6(TwoFactorAnova):
    def _():
        # Also serves for question 12.
        params = {
            "i": 3,
            "j": 5,
            "ssa": 11.7,
            "ssb": 113.5,
            "sse": 25.6,
        }
        # anova = TwoFactorAnova.from_params(params, 0.05)
        anova = TwoFactorAnova.from_params(params, 0.01)
        anova.print()
        print(f"msa - msb: {anova._msa - anova._msb}")

    _()
    return


@app.cell
def example11_7_from_data(TwoFactorAnova):
    def _():
        data = [
            [[0.835, 0.845], [0.822, 0.826], [0.785, 0.795]],
            [[0.855, 0.865], [0.832, 0.836], [0.790, 0.800]],
            [[0.815, 0.825], [0.800, 0.820], [0.770, 0.790]],
        ]
        anova = TwoFactorAnova.from_data(data, 0.10)
        anova.print()

    _()
    return

@app.cell
def example11_7_from_params(TwoFactorAnova):
    def _():
        params = {
            "i": 3,
            "j": 3,
            "k": 2,
            "ssa": 0.0020893333333333306,
            "ssb": 0.008297333333333285,
            "sse": 0.0006659999999999991,
            "ssab": 0.0003253333333333344,
        }
        anova = TwoFactorAnova.from_params(params, 0.05)
        anova.print()
    _()
    return

@app.cell
def example11_9_from_data(TwoFactorAnova):
    def _():
        data = [
            [[13.1, 13.2], [16.3, 15.8], [13.7, 14.3], [15.7, 15.8], [13.5, 12.5]],
            [[15.0, 14.8], [15.7, 16.4], [13.9, 14.3], [13.7, 14.2], [13.4, 13.8]],
            [[14.0, 14.3], [17.2, 16.7], [12.4, 12.3], [14.4, 13.9], [13.2, 13.1]],
        ]
        anova = TwoFactorAnova.from_data(data, 0.05)
        anova.print()
    _()
    return

@app.cell
def problem16(TwoFactorAnova):
    def _():
        i = 3
        j = 4
        k = 3
        ssa = 30763.0
        ssb = 34185.6
        sse = 97436.8
        sst = 205966.6
        ssab = sst - ssa - ssb - sse

        params = {
            "i": i,
            "j": j,
            "k": k,
            "ssa": ssa,
            "ssb": ssb,
            "sse": sse,
            "ssab": ssab,
        }
        anova = TwoFactorAnova.from_params(params, 0.05)
        anova.print()
    _()
    return

@app.cell
def problem18(TwoFactorAnova):
    def _():
        import polars as pl
        data = [
            [[189.7, 188.6, 190.1], [185.1, 179.4, 177.3], [189.0, 193.0, 191.1]],
            [[165.1, 165.9, 167.6], [161.7, 159.8, 161.6], [163.3, 166.6, 170.3]],
        ]
        anova = TwoFactorAnova.from_data(data, 0.05)
        anova.print()

        df = pl.DataFrame(data)
        beta_3 = TwoFactorAnova.fixed_effects_beta(df, 3)
        beta_1 = TwoFactorAnova.fixed_effects_beta(df, 1)
        alpha_1 = TwoFactorAnova.fixed_effects_alpha(df, 1)

        print(f"beta_3 - beta_1 + alpha_1: {beta_3 - beta_1 + alpha_1}")

    _()
    return

@app.cell
def problem22(TwoFactorAnova):
    def _():
        data = [
            [[709, 659], [713, 726], [660, 645]],
            [[668, 685], [722, 740], [692, 720]],
            [[659, 685], [666, 684], [678, 750]],
            [[698, 650], [704, 666], [686, 733]],
        ]
        anova = TwoFactorAnova.from_data(data, 0.05)
        anova.print()
    _()
    return

@app.cell
def problem29():
    def _():
        from scipy.stats import f

        alpha = 0.05

        critical_value = lambda alpha, ndf, ddf: f.ppf(1-alpha, ndf, ddf)
        pvalue = lambda fvalue, ndf, ddf: 1 - f.cdf(fvalue, ndf, ddf)

        a_df = 2
        b_df = 1
        c_df = 3
        ab_df = 2
        ac_df = 6
        bc_df = 3
        abc_df = 6
        e_df = 72

        f_ac_critical = critical_value(alpha, ac_df, e_df)
        f_ac_pvalue = pvalue(1.905, ac_df, e_df)
        print(f"f_ac critical value: {f_ac_critical}")
        print(f"f_ac pvalue: {f_ac_pvalue}")
    _()
    return

@app.cell
def problem30():
    def _():
        critical_value = lambda alpha, ndf, ddf: f.ppf(1-alpha, ndf, ddf)
        pvalue = lambda fvalue, ndf, ddf: 1 - f.cdf(fvalue, ndf, ddf)

        alpha = 0.05

        i = 4
        j = 2
        k = 2
        l = 1

        tdf = i * j * k * l - 1
        adf = i - 1
        bdf = j - 1
        cdf = k - 1
        abdf = (i - 1) * (j - 1)
        acdf = (i - 1) * (k - 1)
        bcdf = (j - 1) * (k - 1)
        abcdf = (i - 1) * (j - 1) * (k - 1)
        edf = i * j * k * (l - 1)

        ssa = 0.22625
        ssb = 0.000025
        ssc = 0.0036
        ssab = 0.004325
        ssac = 0.00065
        ssbc = 0.000625
        ssabc = 0
        sst = 0.2384
        sse = sst - ssa - ssb - ssc - ssab - ssac - ssbc - ssabc

        msa = ssa / adf
        msb = ssb / bdf
        msc = ssc / cdf
        msab = ssab / abdf
        msac = ssac / acdf
        msbc = ssbc / bcdf
        mse = sse / abcdf # "highest order interaction assumed" mse and msabc are assumed equivalent in this example

        f_a = msa / mse
        f_b = msb / mse
        f_c = msc / mse
        f_ab = msab / mse
        f_ac = msac / mse
        f_bc = msbc / mse

        critical_a = critical_value(alpha, adf, abcdf)
        critical_b = critical_value(alpha, bdf, abcdf)
        critical_c = critical_value(alpha, cdf, abcdf)
        critical_ab = critical_value(alpha, abdf, abcdf)
        critical_ac = critical_value(alpha, acdf, abcdf)
        critical_bc = critical_value(alpha, bcdf, abcdf)
        
        print(f"f vs critical (a): {f_a} vs {critical_a}")
        print(f"f vs critical (b): {f_b} vs {critical_b}")
        print(f"f vs critical (c): {f_c} vs {critical_c}")
        print(f"f vs critical (ab): {f_ab} vs {critical_ab}")
        print(f"f vs critical (ac): {f_ac} vs {critical_ac}")
        print(f"f vs critical (bc): {f_bc} vs {critical_bc}")

    _()
    return

if __name__ == "__main__":
    app.run()
