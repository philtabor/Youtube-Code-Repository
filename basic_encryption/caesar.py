from common import alphabet


def translate(message, shift, encrypt=True):
    new_message = ''
    n_chars = len(alphabet)

    for character in message:
        char_idx = alphabet.index(character)
        if encrypt:
            new_char_idx = (char_idx + shift) % n_chars
        elif not encrypt:
            new_char_idx = (char_idx - shift) % n_chars
        new_message += alphabet[new_char_idx]
    return new_message


cipher_shift = 7

print('AB->', translate('AB', cipher_shift))
print('ab->', translate('ab', cipher_shift))
print('Ab->', translate('Ab', cipher_shift))
print('aB->', translate('aB', cipher_shift))

plaintext = 'This is an encrypted message.'
ciphertext = translate(plaintext, cipher_shift, True)
print(plaintext, '->', ciphertext)
original_message = translate(ciphertext, cipher_shift, False)
print(ciphertext, '->', original_message)
