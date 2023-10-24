import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import font_manager, rc

class DataVisualizer:
    @staticmethod
    def visualize(rate, xs: list, ys: list):
        font_test = 'c:/Windows/Fonts/malgun.ttf'
        font_name = font_manager.FontProperties(fname=font_test).get_name()
        rc('font', family=font_name)
        figure: Figure = plt.figure(figsize=(6.5, 6.5))
        ax1 = figure.add_subplot(2, 2, 1, xlabel='emotion', ylabel='count')
        ax2 = figure.add_subplot(2, 2, 2)
        ax3 = figure.add_subplot(2, 2, 3, xlabel='review_rank', ylabel='count')
        x = xs
        y = ys
        ax1.bar(x, y)
        ax2.pie(y, labels=x, autopct='%.1f%%')
        ax3.bar(rate.index, rate.values)
        return figure
    
    @staticmethod
    def save_to_img(file_name, figure):
        figure.savefig(file_name)

