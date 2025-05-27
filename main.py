# main.py
# Реализует генетический алгоритм для выбора клиентов, фильтрацию данных и сохранение результатов.
import pandas as pd
import numpy as np
import utils as EGAUtils
import constants as EConstants
import main_plot

# Параметры алгоритма и банка
pop_size = 60  # Размер популяции
generation_size = 60  # Число поколений
p_xover = 0.8  # Вероятность кроссовера
p_mutation = 0.006  # Вероятность мутации
reproduction_ratio = 0.194  # Вероятность репродукции
num_of_customers = 300  # Число клиентов
bank_required_reserve_ratio = 0.2  # Резервная норма
financial_institutions_deposit = 5000000  # Депозит банка
rD = 0.009  # Ставка по депозитам

# Чтение и фильтрация данных
df_customer = pd.read_csv('customers_test.csv', names=EConstants.get_cols_customer())
# Преобразование в числовой формат
df_customer['Credit Limit'] = pd.to_numeric(df_customer['Credit Limit'], errors='coerce')
df_customer['Loan Size'] = pd.to_numeric(df_customer['Loan Size'], errors='coerce')
df_customer['Loan Age'] = pd.to_numeric(df_customer['Loan Age'], errors='coerce')

valid_ratings = {'AAA', 'AA', 'A', 'BBB', 'BB'}
# Фильтрация корректных записей
df_customer = df_customer[
    (df_customer['Loan Type'].isin(['Mortgage', 'Personal', 'Auto'])) &
    (df_customer['Credit Rating'].isin(valid_ratings)) &
    (~df_customer['Loan Size'].isna()) &
    (~df_customer['Loan Age'].isna()) &
    (~df_customer['Credit Limit'].isna()) &
    ((df_customer['Loan Type'].isin(['Personal', 'Auto']) & (df_customer['Loan Age'] <= 10)) |
     (df_customer['Loan Type'] == 'Mortgage')) &
    (df_customer['Loan Size'] > 0)
]
print("После фильтрации:\n", df_customer)
print("Количество клиентов:", len(df_customer))
excluded = len(pd.read_csv('customers_test.csv')) - len(df_customer)
print(f"Исключено клиентов при фильтрации: {excluded}")

# Корректировка числа клиентов
if len(df_customer) < num_of_customers:
    print(f"Предупреждение: После фильтрации осталось {len(df_customer)} клиентов вместо {num_of_customers}")
    num_of_customers = len(df_customer)

df_customer = df_customer.reset_index(drop=True)

# Инициализация популяции
chromos = EGAUtils.init_population_with_GAMCC(df_customer, num_of_customers, pop_size, bank_required_reserve_ratio,
                                              financial_institutions_deposit)

# Вычисление начального фитнеса
chromos_fitness_vector = \
    EGAUtils.calc_all_fitness_of_chromos(df_customers=df_customer, chromos=chromos, rD=rD,
                                         bank_required_reserve_ratio=bank_required_reserve_ratio,
                                         financial_institutions_deposit=financial_institutions_deposit)

# Основной цикл алгоритма
for generation_index in range(generation_size):
    # Рулеточный отбор
    rated_fitness_vector = EGAUtils.get_rated_fit_vector(chromos_fitness_vector)
    selected_indices = np.random.choice(range(len(chromos)), size=6, p=rated_fitness_vector, replace=False)

    # Обработка пар хромосом
    number_of_worst_chromo_to_be_deleted = 0
    for i in range(0, len(selected_indices), 2):
        if i + 1 < len(selected_indices):
            parent1 = chromos[selected_indices[i], :]
            parent2 = chromos[selected_indices[i + 1], :]
            # Репродукция
            if np.random.random() < reproduction_ratio:
                chromos = np.vstack((chromos, parent1.copy()))
                number_of_worst_chromo_to_be_deleted += 1
            # Кроссовер и мутация
            rand_xover = np.random.uniform(0, 1)
            rand_mutation = np.random.uniform(0, 1)
            xover_ch1 = np.zeros(num_of_customers, dtype=bool)
            xover_ch2 = np.zeros(num_of_customers, dtype=bool)
            mut_ch1 = np.zeros(num_of_customers, dtype=bool)
            mut_ch2 = np.zeros(num_of_customers, dtype=bool)
            is_xovered = False
            is_mutated = False
            if rand_xover <= p_xover:
                xover_ch1, xover_ch2 = EGAUtils.xover(parent1, parent2)
                is_xovered = True
            if rand_mutation <= p_mutation:
                mut_ch1, mut_ch2 = EGAUtils.mutate(parent1, parent2)
                is_mutated = True
            # Добавление потомков от кроссовера
            if is_xovered:
                if EGAUtils.is_GAMCC_satisfied(df_customers=EGAUtils.get_dataframe_by_chromo(df_customer, xover_ch1),
                                               bank_required_reserve_ratio=bank_required_reserve_ratio,
                                               financial_institutions_deposit=financial_institutions_deposit):
                    chromos = np.vstack((chromos, xover_ch1))
                    number_of_worst_chromo_to_be_deleted += 1
                if EGAUtils.is_GAMCC_satisfied(df_customers=EGAUtils.get_dataframe_by_chromo(df_customer, xover_ch2),
                                               bank_required_reserve_ratio=bank_required_reserve_ratio,
                                               financial_institutions_deposit=financial_institutions_deposit):
                    chromos = np.vstack((chromos, xover_ch2))
                    number_of_worst_chromo_to_be_deleted += 1
            # Добавление потомков от мутации
            if is_mutated:
                if EGAUtils.is_GAMCC_satisfied(df_customers=EGAUtils.get_dataframe_by_chromo(df_customer, mut_ch1),
                                               bank_required_reserve_ratio=bank_required_reserve_ratio,
                                               financial_institutions_deposit=financial_institutions_deposit):
                    chromos = np.vstack((chromos, mut_ch1))
                    number_of_worst_chromo_to_be_deleted += 1
                if EGAUtils.is_GAMCC_satisfied(df_customers=EGAUtils.get_dataframe_by_chromo(df_customer, mut_ch2),
                                               bank_required_reserve_ratio=bank_required_reserve_ratio,
                                               financial_institutions_deposit=financial_institutions_deposit):
                    chromos = np.vstack((chromos, mut_ch2))
                    number_of_worst_chromo_to_be_deleted += 1

    # Удаление худших хромосом
    chromos = EGAUtils.delete_chromo_based_on_bad_fit(chromos=chromos, fitness_vector=chromos_fitness_vector,
                                                      number_of_chromo_to_be_deleted=number_of_worst_chromo_to_be_deleted)
    # Обновление фитнеса
    chromos_fitness_vector = EGAUtils.calc_all_fitness_of_chromos(df_customers=df_customer, chromos=chromos, rD=rD,
                                                                  bank_required_reserve_ratio=bank_required_reserve_ratio,
                                                                  financial_institutions_deposit=financial_institutions_deposit)
    print('generation ', generation_index, ' finished with better ', number_of_worst_chromo_to_be_deleted, ' children')

# Поиск лучшего решения
max_found_fit_index = np.argmax(chromos_fitness_vector)
best_solution = chromos[max_found_fit_index, :]
print("best solution is: ", best_solution)
print('These customer has been elected: ')

# Подсчёт статистики
count_of_accepted_customer = 0
count_of_accepted_customer_loan_type_m = 0
count_of_accepted_customer_loan_type_p = 0
count_of_accepted_customer_loan_type_a = 0
D = financial_institutions_deposit
count_of_accepted_customer_credit_rating_aaa = 0
count_of_accepted_customer_credit_rating_aa = 0
count_of_accepted_customer_credit_rating_a = 0
count_of_accepted_customer_credit_rating_bbb = 0
count_of_accepted_customer_credit_rating_bb = 0

# Анализ выбранных клиентов
for i in range(len(best_solution)):
    if best_solution[i]:
        print('customer with index ', i, ' and ID ', i + 1, ' has been elected')
        count_of_accepted_customer += 1
        customer = df_customer.iloc[i, :]
        # Подсчёт по типу кредита
        if customer['Loan Type'] == 'Mortgage':
            count_of_accepted_customer_loan_type_m += 1
        elif customer['Loan Type'] == 'Personal':
            count_of_accepted_customer_loan_type_p += 1
        elif customer['Loan Type'] == 'Auto':
            count_of_accepted_customer_loan_type_a += 1
        # Подсчёт по рейтингу
        if customer['Credit Rating'] == 'AAA':
            count_of_accepted_customer_credit_rating_aaa += 1
        elif customer['Credit Rating'] == 'AA':
            count_of_accepted_customer_credit_rating_aa += 1
        elif customer['Credit Rating'] == 'A':
            count_of_accepted_customer_credit_rating_a += 1
        elif customer['Credit Rating'] == 'BBB':
            count_of_accepted_customer_credit_rating_bbb += 1
        elif customer['Credit Rating'] == 'BB':
            count_of_accepted_customer_credit_rating_bb += 1

# Вывод метрик
best_df = df_customer.iloc[np.where(best_solution)[0], :]
print("Best fitness:", chromos_fitness_vector[max_found_fit_index])
print("Sum of Loan Size:", best_df['Loan Size'].sum())
print("Expected Loss:", EGAUtils.calc_sum_of_landa(best_df))

# Сохранение результатов
df_results = pd.read_csv('results.txt', names=EConstants.get_cols_result())
df_results = pd.concat([df_results, pd.DataFrame([[
    count_of_accepted_customer_loan_type_m / count_of_accepted_customer if count_of_accepted_customer > 0 else 0,
    count_of_accepted_customer_loan_type_p / count_of_accepted_customer if count_of_accepted_customer > 0 else 0,
    count_of_accepted_customer_loan_type_a / count_of_accepted_customer if count_of_accepted_customer > 0 else 0,
    D,
    pop_size,
    count_of_accepted_customer_credit_rating_aaa / count_of_accepted_customer if count_of_accepted_customer > 0 else 0,
    count_of_accepted_customer_credit_rating_aa / count_of_accepted_customer if count_of_accepted_customer > 0 else 0,
    count_of_accepted_customer_credit_rating_a / count_of_accepted_customer if count_of_accepted_customer > 0 else 0,
    count_of_accepted_customer_credit_rating_bbb / count_of_accepted_customer if count_of_accepted_customer > 0 else 0,
    count_of_accepted_customer_credit_rating_bb / count_of_accepted_customer if count_of_accepted_customer > 0 else 0,
    count_of_accepted_customer,
    generation_size]], columns=df_results.columns)], ignore_index=True)

df_results.to_csv('results.txt', index=False, header=False)
main_plot.main()
print('finish')