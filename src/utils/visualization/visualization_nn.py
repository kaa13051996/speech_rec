from matplotlib import pyplot as plt
from src.utils.save import save_plots, save_structure_network
from src.const.paths import PLOTS_DIR


def visualization_acc(history):
    plt.figure(num=1, dpi=80, figsize=(512 / 80, 384 / 80))
    label = 'acc'
    main(plt, history, label)
    # plt.plot(history['acc'], label='Точность на обучающем наборе')
    # plt.plot(history['val_acc'], label='Точность на проверочном наборе')
    # plt.xlabel('Эпоха обучения')
    # plt.ylabel('Точность')
    # plt.legend()
    # plt.show()
    return plt.figure(1)


def visualization_loss(history):
    plt.figure(num=2, dpi=80, figsize=(512 / 80, 384 / 80))
    label = 'loss'
    main(plt, history, label)
    # plt.plot(history['loss'], label='Ошибка на обучающем наборе')
    # plt.plot(history['val_loss'], label='Ошибка на проверочном наборе')
    # plt.xlabel('Эпоха обучения')
    # plt.ylabel('Ошибка')
    # plt.legend()
    # plt.show()
    return plt.figure(2)


def main(plt, history, label):
    # dpi = 80
    # fig = plt.figure(dpi=dpi, figsize=(512 / dpi, 384 / dpi))
    ax = plt.axes()
    ax.yaxis.grid(True, zorder=1)

    epoches_names = [epoch for epoch in range(len(history[label]))]
    train = [score for score in history[label]]
    test = [score for score in history[f'val_{label}']]

    xs = range(len(epoches_names))

    plt.bar([x + 0.05 for x in xs], train,
            width=0.2, color='red', alpha=0.7, label=label,
            zorder=2)
    plt.bar([x + 0.3 for x in xs], test,
            width=0.2, color='blue', alpha=0.7, label=f'val_{label}',
            zorder=2)
    plt.xticks(xs, epoches_names)
    plt.legend(loc='lower right')
    plt.ylim(min(train+test)-0.02, max(train+test)+0.02)


def run(history, is_save=True, network=None):
    plt_acc = visualization_acc(history)
    plt_loss = visualization_loss(history)
    if is_save:
        save_plots(plt_acc, dir=PLOTS_DIR, name='acc_hist')
        save_plots(plt_loss, dir=PLOTS_DIR, name='loss_hist')
        save_structure_network(network) if network is not None else None
