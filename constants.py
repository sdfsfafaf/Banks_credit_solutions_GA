def get_cols_customer() -> list():
    return ['ID', 'Loan Age', 'Loan Size', 'Loan Type', 'Credit Rating', 'Credit Limit']


def get_cols_bank() -> list():
    return ['Loan Interest Rate', 'Expected Loan Loss', 'Deposit Rate', 'Reserve Ratio', 'Transaction Cost']


def get_cols_result() -> list():
    return ['M%', 'P%', 'LA%', 'D', 'POP_SIZE', 'AAA%', 'AA%', 'A%', 'BBB%', 'BB%', 'ACCEPTED_CUSTOMERS',
            'GENERATION_SIZE']