import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from typing import List

    class TwoFactorAnova:
        def __init__(
            self,
            alpha: float = 0.05,
            data: List[List[float]] = None,
            params: dict[str, any] = {},
        ):
            self.alpha = alpha
            if data:
                self.data = pl.DataFrame(data)
            #####################################
            # I and J
            #####################################
            self.i: int = (
                (data and self.__class__.i(self.data))
                or params.get("i")
                or print("no value assigned for i")
            )
            self.j: int = (
                (data and self.__class__.j(self.data))
                or params.get("j")
                or print("no value assigned for j")
            )
            #####################################
            # Degrees of freedom
            #####################################
            # Factor A df
            self.adf: int = (
                (data and self.__class__.adf(self.data))
                or params.get("adf")
                or print("no value assigned for adf")
            )
            # Factor B df
            self.bdf: int = (
                (data and self.__class__.bdf(self.data))
                or params.get("bdf")
                or print("no value assigned for bdf")
            )
            # Error df
            self.edf: int = (
                (data and self.__class__.edf(self.data))
                or params.get("edf")
                or print("no value assigned for edf")
            )
            # Total df
            self.tdf: int = (
                (data and self.__class__.tdf(self.data))
                or params.get("tdf")
                or print("no value assigned for tdf")
            )
            #####################################
            # Sum of Squares
            #####################################
            # Factor A sum of squares
            self.ssa: float = (
                (data and self.__class__.ssa(self.data))
                or params.get("ssa")
                or print("no value assigned for ssa")
            )
            # Factor B sum of squares
            self.ssb: float = (
                (data and self.__class__.ssb(self.data))
                or params.get("ssb")
                or print("no value assigned for ssb")
            )
            # Error sum of squares
            self.sse: float = (
                (data and self.__class__.sse(self.data))
                or params.get("sse")
                or print("no value assigned for sse")
            )
            # Total sum of squares
            self.sst: float = (
                (data and self.__class__.sst(self.data))
                or params.get("sst")
                or print("no value assigned for sst")
            )
            #####################################
            # Mean Squares
            #####################################
            # Factor A mean square
            self.msa: float = (
                (data and self.__class__.msa(self.data))
                or params.get("msa")
                or print("no value assigned for msa")
            )
            # Factor B mean square
            self.msb: float = (
                (data and self.__class__.msb(self.data))
                or params.get("msb")
                or print("no value assigned for msb")
            )
            # Error mean square
            self.mse: float = (
                (data and self.__class__.mse(self.data))
                or params.get("mse")
                or print("no value assigned for mse")
            )
            #####################################
            # F test statistic
            #####################################
            # f test statistic of factor A
            self.f_a: float = (
                (data and self.__class__.f_a(self.data))
                or params.get("f_a")
                or print("no value assigned for f_a")
            )
            # f test statistic of factor B
            self.f_b: float = (
                (data and self.__class__.f_b(self.data))
                or params.get("f_b")
                or print("no value assigned for f_b")
            )

            if data is None:
                return

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
            print(f"i: {self.i}")
            print(f"j: {self.j}")
            print("-----------------------------------")
            print(f"Factor A df: {self.adf}")
            print(f"Factor B df: {self.bdf}")
            print(f"Error df: {self.edf}")
            print(f"Total df: {self.tdf}")
            print("-----------------------------------")
            print(f"Factor A sum of squares: {self.ssa}")
            print(f"Factor B sum of squares: {self.ssb}")
            print(f"Error sum of squares: {self.sse}")
            print(f"Total sum of squares: {self.sst}")
            print("-----------------------------------")
            print(f"Factor A mean square: {self.msa}")
            print(f"Factor B mean square: {self.msb}")
            print(f"Error mean square: {self.mse}")
            print("-----------------------------------")
            print(f"alpha: {self.alpha}")
            print(f"f test statistic of factor A: {self.f_a}")
            print(f"f test statistic of factor B: {self.f_b}")
            print("-----------------------------------")

        @staticmethod
        def i(data: pl.DataFrame):
            return data.shape[1]

        @staticmethod
        def j(data: pl.DataFrame):
            return data.shape[0]

        @staticmethod
        def adf(data: pl.DataFrame):
            i = TwoFactorAnova.i(data)
            return i - 1

        @staticmethod
        def bdf(data: pl.DataFrame):
            j = TwoFactorAnova.j(data)
            return j - 1

        @staticmethod
        def edf(data: pl.DataFrame):
            _cls = TwoFactorAnova
            return _cls.adf(data) * _cls.bdf(data)

        @staticmethod
        def tdf(data: pl.DataFrame):
            _cls = TwoFactorAnova
            return _cls.i(data) * _cls.j(data) - 1

        @staticmethod
        def grand_mean(data: pl.DataFrame):
            return data.to_numpy().mean()

        @staticmethod
        def ssa(data: pl.DataFrame):
            _cls = TwoFactorAnova
            grand_mean = _cls.grand_mean(data)
            return (
                _cls.j(data)
                * data.mean()
                .select(pl.all().map_elements(lambda x: pow(x - grand_mean, 2)))
                .to_numpy()
                .sum()
            )

        @staticmethod
        def ssb(data: pl.DataFrame):
            _cls = TwoFactorAnova
            grand_mean = _cls.grand_mean(data)
            return (
                _cls.i(data)
                * data.transpose()
                .mean()
                .select(pl.all().map_elements(lambda x: pow(x - grand_mean, 2)))
                .to_numpy()
                .sum()
            )

        @staticmethod
        def sse(data: pl.DataFrame):
            result = 0
            x_i = data.mean().to_numpy()[0]
            x_j = data.transpose().mean().to_numpy()[0]
            nd = data.transpose().to_numpy()
            grand_mean = TwoFactorAnova.grand_mean(data)
            for i in range(nd.shape[0]):
                for j in range(nd.shape[1]):
                    result += pow(nd[i][j] - x_i[i] - x_j[j] + grand_mean, 2)
            return result

        @staticmethod
        def sst(data: pl.DataFrame):
            grand_mean = TwoFactorAnova.grand_mean(data)
            return data.select(pl.all().map_elements(lambda x: pow(x - grand_mean, 2))).to_numpy().sum()

        @staticmethod
        def msa(data: pl.DataFrame):
            _cls = TwoFactorAnova
            return _cls.ssa(data) / _cls.adf(data)

        @staticmethod
        def msb(data: pl.DataFrame):
            _cls = TwoFactorAnova
            return _cls.ssb(data) / _cls.bdf(data)

        @staticmethod
        def mse(data: pl.DataFrame):
            _cls = TwoFactorAnova
            return _cls.sse(data) / _cls.edf(data)

        @staticmethod
        def f_a(data: pl.DataFrame):
            _cls = TwoFactorAnova
            return _cls.msa(data) / _cls.mse(data)
        
        def f_b(data: pl.DataFrame):
            _cls = TwoFactorAnova
            return _cls.msb(data) / _cls.mse(data)

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


if __name__ == "__main__":
    app.run()
