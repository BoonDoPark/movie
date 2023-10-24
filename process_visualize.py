from visualize import DataVisualizer

class MovieVisualize:
    @staticmethod
    def run(rate, xs: list, ys: list, file_name: str):
        figure = DataVisualizer.visualize(rate, xs, ys)
        DataVisualizer.save_to_img(file_name, figure)

