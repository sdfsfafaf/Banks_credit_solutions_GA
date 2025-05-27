# constants.py
# Определяет названия столбцов для данных.
def get_cols_customer() -> list:
    # Столбцы для данных клиентов
    return ['ID', 'Loan Age', 'Loan Size', 'Loan Type', 'Credit Rating', 'Credit Limit']

def get_cols_bank() -> list:
    # Столбцы для банковских параметров
    return ['Loan Interest Rate', 'Expected Loan Loss', 'Deposit Rate', 'Reserve Ratio', 'Transaction Cost']

def get_cols_result() -> list:
    # Столбцы для результатов
    return ['M%', 'P%', 'LA%', 'D', 'POP_SIZE', 'AAA%', 'AA%', 'A%', 'BBB%', 'BB%', 'ACCEPTED_CUSTOMERS', 'GENERATION_SIZE']