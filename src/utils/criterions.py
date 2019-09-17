from scipy import stats as scipystats
from src.const.paths import BALANCE_FOR_STAT
from csv import reader
from numpy import array


def read_data(BALANCE_FOR_STAT):
    result = []
    with open(BALANCE_FOR_STAT, newline='') as csvfile:
        csv = reader(csvfile, delimiter=';')
        for row in csv:
            result.append(list(map(float, row)))
    return array(result)


def student_t(data_1, data_2):
    # print(f'{data_1[0]}\t{data_2[0]}')
    return scipystats.ttest_ind(data_1, data_2, equal_var=True)


def run():
    result = []
    data = read_data(BALANCE_FOR_STAT)
    for cikl in range(data.shape[1]):
        temp = []
        for column in range(data.shape[1]):
            if cikl != column:
                temp.append(student_t(data[:, cikl], data[:, column]))
        result.append(temp)
    return array(result)


if __name__ == '__main__':
    run()
    # rvs1 = scipystats.norm.rvs(loc=5, scale=10, size=500)
    # rvs2 = scipystats.norm.rvs(loc=5, scale=10, size=500)
    # print(student_t(rvs1, rvs2))
