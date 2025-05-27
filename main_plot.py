# visualization.py
# Генерирует графики для анализа результатов из results.csv.
import pandas as pd
import numpy as np
import utils as EGAUtils
import constants as EConstants
import matplotlib.pyplot as plt

# Чтение и обработка результатов
def main():

    df_results = pd.read_csv('results.txt', names=EConstants.get_cols_result())
# Преобразование долей в проценты
    df_results['M%'] = (df_results['M%'] * 100).round(2)
    df_results['P%'] = (df_results['P%'] * 100).round(2)
    df_results['LA%'] = (df_results['LA%'] * 100).round(2)
    df_results['AAA%'] = (df_results['AAA%'] * 100).round(2)
    df_results['AA%'] = (df_results['AA%'] * 100).round(2)
    df_results['A%'] = (df_results['A%'] * 100).round(2)
    df_results['BBB%'] = (df_results['BBB%'] * 100).round(2)
    df_results['BB%'] = (df_results['BB%'] * 100).round(2)

# График типов кредитов
    df_results[['M%', 'P%', 'LA%']].plot()
    plt.xlabel('Итерация')
    plt.ylabel('Процент (%)')
    plt.title('Распределение типов кредитов')
    plt.legend(['Mortgage', 'Personal', 'Auto'])
    plt.grid(True)  # Сетка для читаемости
    plt.savefig('loan_types.png')  # Сохранение в файл
    plt.show()

# График рейтингов
    df_results[['AAA%', 'AA%', 'A%', 'BBB%', 'BB%']].plot()
    plt.xlabel('Итерация')
    plt.ylabel('Процент (%)')
    plt.title('Распределение кредитных рейтингов')
    plt.legend(['AAA', 'AA', 'A', 'BBB', 'BB'])
    plt.grid(True)
    plt.savefig('ratings.png')
    plt.show()