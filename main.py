import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def example12_1():
    def _():
        import matplotlib.pyplot as plt

        x = [0.40, 0.42, 0.48, 0.51, 0.57, 0.60, 0.70, 0.75, 0.75, 0.78, 0.84, 0.95, 0.99, 1.03, 1.12, 1.15, 1.20, 1.25, 1.25, 1.28, 1.30, 1.34, 1.37, 1.40, 1.43, 1.46, 1.49, 1.55, 1.58, 1.60]
        y = [1.02, 1.21, 0.88, 0.98, 1.52, 1.83, 1.50, 1.80, 1.74, 1.63, 2.00, 2.80, 2.48, 2.47, 3.05, 3.18, 3.76, 3.68, 3.82, 3.21, 4.27, 3.12, 3.99, 3.75, 4.10, 4.18, 3.77, 4.34, 4.21, 4.92]

        plt.scatter(x, y)
        plt.xlabel('width of palprebal fissure')
        plt.ylabel('ocular surface area')
        plt.title('ocular surface area vs. width of parprebal fissure')
        plt.show()

    _()
    return

if __name__ == "__main__":
    app.run()
