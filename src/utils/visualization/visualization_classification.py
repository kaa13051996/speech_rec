from matplotlib import pyplot as plt
from numpy import arange


def error_visualization(scores_train, scores_test, metric, title, count_speaker=20):
    speakers = [f'#{speaker+1}' for speaker in range(count_speaker)]
    ind = arange(len(scores_train))  # the x locations for the groups
    width = 0.35
    fig, ax = plt.subplots()
    ax.set_xticks(ind)
    ax.set_xticklabels(speakers)
    rects1 = ax.bar(ind - width / 2, scores_train, width,
                    color='SkyBlue', label='Train')
    rects2 = ax.bar(ind + width / 2, scores_test, width,
                    color='IndianRed', label='Test')

    # rows = ['Train', 'Test']
    # the_table = ax.table(cellText=[scores_train, scores_test], rowLabels=rows)
    plt.ylim(0.9, 1)
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('Дикторы')
    plt.legend()
    # plt.show()
    return plt
