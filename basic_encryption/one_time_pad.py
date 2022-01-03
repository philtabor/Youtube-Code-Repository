from common import alphabet, generate_one_time_pad


def translate(message, one_time_pad, encrypt=True):
    new_message = ''

    n_chars = len(alphabet)

    for src, key in zip(message, one_time_pad):
        char_idx = alphabet.index(src)
        pad_idx = alphabet.index(key)
        if encrypt:
            new_char_idx = (char_idx + pad_idx) % n_chars
        elif not encrypt:
            new_char_idx = (char_idx - pad_idx) % n_chars
        new_message += alphabet[new_char_idx]

    return new_message


message = 'This is an encrypted message.'
secret_key = generate_one_time_pad(len(message), alphabet)
encrypted_message = translate(message, secret_key, True)
original_message = translate(encrypted_message, secret_key, False)

print(message, '->', encrypted_message)
print(encrypted_message, '->', original_message)
