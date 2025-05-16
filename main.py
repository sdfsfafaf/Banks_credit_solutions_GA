import pandas as pd
import numpy as np
import utils as EGAUtils
import constants as EConstants

pop_size = 20  # chromos population
generation_size = 70
p_xover = 0.8
p_mutation = 0.05
reproduction_ratio = 0.194
num_of_customers = 15  # len of each chromo

bank_required_reserve_ratio = 0  # K
financial_institutions_deposit = 45000  # D
rD = 0.009

pi = 2  # count of best chromo to be selected for operations[xover, mute]

df_customer = pd.read_csv('customers_test.csv', names=EConstants.get_cols_customer())
# 'ID', 'Loan Age', 'Loan Size', 'Loan Type', 'Credit Rating', 'Credit Limit'
# df_customer['ID'] = pd.to_numeric(df_customer['ID'])
df_customer['Credit Limit'] = pd.to_numeric(df_customer['Credit Limit'])
df_customer['Loan Size'] = pd.to_numeric(df_customer['Loan Size'])
df_customer['Loan Age'] = pd.to_numeric(df_customer['Loan Age'])
print(df_customer)

chromos = EGAUtils.init_population_with_GAMCC(df_customer, num_of_customers, pop_size, bank_required_reserve_ratio,
                                              financial_institutions_deposit)

print(chromos.shape)

chromos_fitness_vector = \
    EGAUtils.calc_all_fitness_of_chromos(df_customers=df_customer, chromos=chromos, rD=rD,
                                         bank_required_reserve_ratio=bank_required_reserve_ratio,
                                         financial_institutions_deposit=financial_institutions_deposit)
print(chromos_fitness_vector.shape)
print(chromos_fitness_vector)

sum_of_all_fitness = chromos_fitness_vector.sum()

print(sum_of_all_fitness)

rated_fitness_vector = EGAUtils.get_rated_fit_vector(chromos_fitness_vector)
print(rated_fitness_vector)
print(rated_fitness_vector.shape)
print(len(rated_fitness_vector))

for generation_index in range(generation_size):
    selected_chromos = []
    while len(selected_chromos) < pi:
        max_checked_val = 0
        max_found_index = 0
        for chromo_index in range(len(rated_fitness_vector)):
            if chromo_index in selected_chromos:
                continue
            if max_checked_val < rated_fitness_vector[chromo_index]:
                max_checked_val = rated_fitness_vector[chromo_index]
                max_found_index = chromo_index
        selected_chromos.append(max_found_index)
        # print('chromo ', max_found_index, ' selected for operations')

    parent1 = chromos[selected_chromos[0], :]
    parent2 = chromos[selected_chromos[1], :]
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

    number_of_worst_chromo_to_be_deleted = 0
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

    rated_fitness_vector = EGAUtils.get_rated_fit_vector(chromos_fitness_vector)

    chromos = EGAUtils.delete_chromo_based_on_bad_fit(chromos=chromos, fitness_vector=chromos_fitness_vector,
                                                      number_of_chromo_to_be_deleted=number_of_worst_chromo_to_be_deleted)
    # print(chromos.shape)
    print('generation ', generation_index, ' finished with better ', number_of_worst_chromo_to_be_deleted, ' children')

print("\n\n###############\n\n")

max_found_fit_index = 0
max_found_fit_val = 0

while True:
    for i in range(len(chromos_fitness_vector)):
        if max_checked_val < chromos_fitness_vector[i]:
            max_checked_val = chromos_fitness_vector[i]
            max_found_fit_index = i
    if True in chromos[max_found_fit_index, :]:
        break

best_solution = chromos[max_found_fit_index, :]
print("best solution is: ", best_solution)
print('These customer has been elected: ')

count_of_accepted_customer = 0
# Loan Type
count_of_accepted_customer_loan_type_m = 0
count_of_accepted_customer_loan_type_p = 0
count_of_accepted_customer_loan_type_a = 0
D = financial_institutions_deposit
# Credit Rating
count_of_accepted_customer_credit_rating_aaa = 0
count_of_accepted_customer_credit_rating_aa = 0
count_of_accepted_customer_credit_rating_a = 0
count_of_accepted_customer_credit_rating_bbb = 0
count_of_accepted_customer_credit_rating_bb = 0

for i in range(len(best_solution)):
    if best_solution[i]:
        print('customer with index ', i, ' and ID ', i + 1, ' has been elected')
        count_of_accepted_customer += 1
        customer = df_customer.iloc[i, :]
        # 'ID', 'Loan Age', 'Loan Size', 'Loan Type', 'Credit Rating', 'Credit Limit'
        if customer['Loan Type'] == 'Mortgage':
            count_of_accepted_customer_loan_type_m += 1
        elif customer['Loan Type'] == 'Personal':
            count_of_accepted_customer_loan_type_p += 1
        elif customer['Loan Type'] == 'Auto':
            count_of_accepted_customer_loan_type_a += 1

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

df_results = pd.read_csv('results.csv', names=EConstants.get_cols_result())
df_results = pd.concat([df_results, pd.DataFrame([[
    count_of_accepted_customer_loan_type_m / count_of_accepted_customer,
    count_of_accepted_customer_loan_type_p / count_of_accepted_customer,
    count_of_accepted_customer_loan_type_a / count_of_accepted_customer,
    D,
    pop_size,
    count_of_accepted_customer_credit_rating_aaa / count_of_accepted_customer,
    count_of_accepted_customer_credit_rating_aa / count_of_accepted_customer,
    count_of_accepted_customer_credit_rating_a / count_of_accepted_customer,
    count_of_accepted_customer_credit_rating_bbb / count_of_accepted_customer,
    count_of_accepted_customer_credit_rating_bb / count_of_accepted_customer,
    count_of_accepted_customer,
    generation_size]], columns=df_results.columns)], ignore_index=True)

df_results.to_csv('results.csv', index=False, header=False)

print('finish')