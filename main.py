import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from typing import List

    class TwoFactorAnova:
        data_key = "data"

        def __init__(
            self,
            alpha: float = 0.05,
            data: List[List[float]] = None,
            params: dict[str, any] = {},
        ):
            self.alpha = alpha
            if data:
                self.data = pl.DataFrame({
                    self.__class__.data_key: data,
                })
            #####################################
            # I and J
            #####################################
            self.i: int = (
                (data and self.__class__._i(self.data))
                or params.get("i")
                or print("no value assigned for i")
            )
            self.j: int = (
                (data and print("j not implemented"))
                or params.get("j")
                or print("no value assigned for j")
            )
            #####################################
            # Degrees of freedom
            #####################################
            # Factor A df
            self.adf: int = (
                (data and print("adf not implemented"))
                or params.get("adf")
                or print("no value assigned for adf")
            )
            # Factor B df
            self.bdf: int = (
                (data and print("bdf not implemented"))
                or params.get("bdf")
                or print("no value assigned for bdf")
            )
            # Error df
            self.edf: int = (
                (data and print("edf not implemented"))
                or params.get("edf")
                or print("no value assigned for edf")
            )
            #####################################
            # Sum of Squares
            #####################################
            # Factor A sum of squares
            self.ssa: float = (
                (data and print("ssa not implemented"))
                or params.get("ssa")
                or print("no value assigned for ssa")
            )
            # Factor B sum of squares
            self.ssb: float = (
                (data and print("ssb not implemented"))
                or params.get("ssb")
                or print("no value assigned for ssb")
            )
            # Error sum of squares
            self.sse: float = (
                (data and print("sse not implemented"))
                or params.get("sse")
                or print("no value assigned for sse")
            )
            # Total sum of squares
            self.sst: float = (
                (data and print("sst not implemented"))
                or params.get("sst")
                or print("no value assigned for sst")
            )
            #####################################
            # Mean Squares
            #####################################
            # Factor A mean square
            self.msa: float = (
                (data and print("msa not implemented"))
                or params.get("msa")
                or print("no value assigned for msa")
            )
            # Factor B mean square
            self.msb: float = (
                (data and print("msb not implemented"))
                or params.get("msb")
                or print("no value assigned for msb")
            )
            # Error mean square
            self.mse: float = (
                (data and print("mse not implemented"))
                or params.get("mse")
                or print("no value assigned for mse")
            )
            #####################################
            # F test statistic
            #####################################
            # f test statistic of factor A
            self.f_a: float = (
                (data and print("f_a not implemented"))
                or params.get("f_a")
                or print("no value assigned for f_a")
            )
            # f test statistic of factor B
            self.f_b: float = (
                (data and print("f_b not implemented"))
                or params.get("f_b")
                or print("no value assigned for f_b")
            )

            if data is None:
                return

        @classmethod
        def from_data(cls, data: List[List[float]], alpha: float):
            assert(data is not None)
            assert(alpha is not None)
            return cls(
                alpha,
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
        def _i(data: pl.DataFrame):
            pass
            


    return (TwoFactorAnova,)


@app.cell
def _(TwoFactorAnova):
    data = [
        [0.97, 0.48, 0.48, 0.46],
        [0.77, 0.14, 0.22, 0.25],
        [0.67, 0.39, 0.57, 0.19],
    ]
    anova = TwoFactorAnova.from_data(data, )
    anova.print()
    return


if __name__ == "__main__":
    app.run()
