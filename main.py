import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
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
                self._data = pl.DataFrame(data)
            #####################################
            # I and J
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
            # Total sum of squares
            self._sst: float = (
                (data and self.__class__.data_to_sst(self._data))
                or params.get("sst")
                or print("no value assigned for sst")
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
            pass

        def print(self):
            print("-----------------------------------")
            print(f"i: {self._i}")
            print(f"j: {self._j}")
            print("-----------------------------------")
            print(f"Factor A df: {self._adf}")
            print(f"Factor B df: {self._bdf}")
            print(f"Error df: {self._edf}")
            print(f"Total df: {self._tdf}")
            print("-----------------------------------")
            print(f"Factor A sum of squares: {self._ssa}")
            print(f"Factor B sum of squares: {self._ssb}")
            print(f"Error sum of squares: {self._sse}")
            print(f"Total sum of squares: {self._sst}")
            print("-----------------------------------")
            print(f"Factor A mean square: {self._msa}")
            print(f"Factor B mean square: {self._msb}")
            print(f"Error mean square: {self._mse}")
            print("-----------------------------------")
            print(f"alpha: {self._alpha}")
            print(f"f test statistic of factor A: {self._f_a}")
            print(f"f_a critical value: {self._f_a_critical_value}")
            print(f"f_a p-value: {self._f_a_pvalue}")
            print(f"f test statistic of factor B: {self._f_b}")
            print(f"f_b critical value: {self._f_b_critical_value}")
            print(f"f_b p-value: {self._f_b_pvalue}")
            print("-----------------------------------")

        @classmethod
        def i(cls, data: pl.DataFrame):
            return data.shape[1]

        @classmethod
        def j(cls, data: pl.DataFrame):
            return data.shape[0]

        @classmethod
        def data_to_adf(cls, data: pl.DataFrame):
            i = cls.i(data)
            return i - 1

        def adf(self):
            return self._i - 1

        @classmethod
        def data_to_bdf(cls, data: pl.DataFrame):
            j = cls.j(data)
            return j - 1

        def bdf(self):
            return self._j - 1

        @classmethod
        def data_to_edf(cls, data: pl.DataFrame):
            return cls.data_to_adf(data) * cls.data_to_bdf(data)

        def edf(self):
            return self._adf * self._bdf

        @classmethod
        def data_to_tdf(cls, data: pl.DataFrame):
            return cls.i(data) * cls.j(data) - 1

        def tdf(self):
            return self._i * self._j - 1

        @classmethod
        def grand_mean(cls, data: pl.DataFrame):
            return data.to_numpy().mean()

        @classmethod
        def data_to_ssa(cls, data: pl.DataFrame):
            grand_mean = cls.grand_mean(data)
            return (
                cls.j(data)
                * data.mean()
                .select(pl.all().map_elements(lambda x: pow(x - grand_mean, 2)))
                .to_numpy()
                .sum()
            )

        @classmethod
        def data_to_ssb(cls, data: pl.DataFrame):
            grand_mean = cls.grand_mean(data)
            return (
                cls.i(data)
                * data.transpose()
                .mean()
                .select(pl.all().map_elements(lambda x: pow(x - grand_mean, 2)))
                .to_numpy()
                .sum()
            )

        @classmethod
        def data_to_sse(cls, data: pl.DataFrame):
            result = 0
            x_i = data.mean().to_numpy()[0]
            x_j = data.transpose().mean().to_numpy()[0]
            nd = data.transpose().to_numpy()
            grand_mean = TwoFactorAnova.grand_mean(data)
            for i in range(nd.shape[0]):
                for j in range(nd.shape[1]):
                    result += pow(nd[i][j] - x_i[i] - x_j[j] + grand_mean, 2)
            return result

        @classmethod
        def data_to_sst(cls, data: pl.DataFrame):
            grand_mean = TwoFactorAnova.grand_mean(data)
            return (
                data.select(pl.all().map_elements(lambda x: pow(x - grand_mean, 2)))
                .to_numpy()
                .sum()
            )

        def msa(self):
            return self._ssa / self._adf

        def msb(self):
            return self._ssb / self._bdf

        def mse(self):
            return self._sse / self._edf

        def f_a(self):
            return self._msa / self._mse

        def f_b(self):
            return self._msb / self._mse

        @classmethod
        def fixed_effects_alpha(cls, data: pl.DataFrame, i):
            return data.transpose().to_numpy()[i-1].mean() - cls.grand_mean(data)

        @classmethod
        def fixed_effects_beta(cls, data: pl.DataFrame, j):
            return data.to_numpy()[j-1].mean() - cls.grand_mean(data)

    return (TwoFactorAnova,)


@app.cell
def _(TwoFactorAnova):
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

@app.cell
def _(TwoFactorAnova):
    def _():
        import polars as pl
        data = pl.DataFrame([
            [2, 5],
            [3, 6],
        ])
        print(TwoFactorAnova.fixed_effects_alpha(data, 1))
        print(TwoFactorAnova.fixed_effects_alpha(data, 2))
        print(TwoFactorAnova.fixed_effects_beta(data, 1))
        print(TwoFactorAnova.fixed_effects_beta(data, 2))

    _()
    return

if __name__ == "__main__":
    app.run()
