from math import floor


class Options:
    def __init__(self, all_observation, epochs=10, batch_size=None, steps=None):
        self.all_observation = self.check_all_observation(all_observation)
        self.epochs = self.check_epochs(epochs)
        self.steps = self.step_calculation(steps, batch_size)
        self.batch_size = self.batch_calculation(batch_size)
        self.init_interval = [0, self.batch_size]
        # self.check_options()

    def check_options(self):
        if self.batch_size * self.steps * self.epochs > self.all_observation:
            raise Exception('Заданные параметры превышают общее количество наблюдений!')

    @staticmethod
    def check_all_observation(all_observation):
        if not check_none(all_observation) and check_positive(all_observation) and check_type_int(all_observation):
            return all_observation
        else:
            raise Exception('Ошибка инициализация параметра - количество всех наблюдений.')

    @staticmethod
    def check_epochs(epochs):
        if not check_none(epochs) and check_positive(epochs) and check_type_int(epochs):
            return epochs
        else:
            raise Exception('Ошибка инициализация параметра - количество эпох.')

    def step_calculation(self, steps, batch_size):
        if not check_none(steps) and check_positive(steps) and check_type_int(steps):
            return steps
        else:
            if not check_none(batch_size) and check_positive(batch_size) and check_type_int(batch_size):
                # return floor(self.all_observation / (batch_size * self.epochs))
                return floor(self.all_observation / batch_size)
            else:
                raise Exception(
                    f'Заданное steps ({steps}) не подходит, также нельзя рассчитать - из-за batch_size ({batch_size}).')

    def batch_calculation(self, batch_size):
        if not check_none(batch_size) and check_positive(batch_size) and check_type_int(batch_size):
            return batch_size
        else:
            return (floor(self.all_observation / (self.steps * self.epochs)))


def check_type_int(obj):
    if type(obj) is not int:
        # print(f'Число {obj} не является int.')
        return False
    else:
        return True


def check_positive(obj):
    if obj < 1:
        # print(f'Число {obj} не является положительным.')
        return False
    else:
        return True


def check_none(obj):
    if obj is None:
        # print('Переменная является None.')
        return True
    else:
        return False


batch_size = 100
TRAIN = Options(
    all_observation=1800,
    batch_size=batch_size
)

VAL = Options(
    all_observation=500,
    batch_size=batch_size
)

TEST = Options(
    all_observation=200,
    batch_size=batch_size
)

# TRAIN = Options(
#     all_observation=COUNT_SPEAKERS * (COUNT_RECORDS_TRAINING - COUNT_RECORDS_VAL),
#     batch_size=batch_size
# )
#
# VAL = Options(
#     all_observation=COUNT_SPEAKERS * COUNT_RECORDS_VAL,
#     batch_size=batch_size
# )
#
# TEST = Options(
#     all_observation=COUNT_SPEAKERS * COUNT_RECORDS_TEST,
#     batch_size=batch_size
# )
