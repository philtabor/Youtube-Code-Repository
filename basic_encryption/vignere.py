from common import alphabet, generate_one_time_pad


def make_vignere_table():
    table = [['']] * len(alphabet)
    for idx, character in enumerate(alphabet):
        row = []
        for char in alphabet[idx:]:
            row.append(char)
        for char in alphabet[:idx]:
            row.append(char)
        table[idx] = row
    return table


def translate(message, vig_table, one_time_pad, encrypt=True):
    new_message = ''

    if encrypt:
        for src, key in zip(message, one_time_pad):
            row = vig_table[:][0].index(key)
            col = vig_table[0][:].index(src)
            new_message += vig_table[row][col]
    elif not encrypt:
        for src, key in zip(message, one_time_pad):
            row = vig_table[:][0].index(key)
            col = vig_table[row][:].index(src)
            new_message += vig_table[0][col]
    return new_message


table = make_vignere_table()
message = 'This is an encrypted message.'
secret_key = generate_one_time_pad(len(message), alphabet)
encrypted_message = translate(message, table, secret_key, True)
original_message = translate(encrypted_message, table, secret_key, False)

print(message, '->', encrypted_message)
print(encrypted_message, '->', original_message)
