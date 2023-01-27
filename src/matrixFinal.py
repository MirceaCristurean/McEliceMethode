'''
This file contains the implementation of encrypt and decrypt a message using 2 methods: McEliece cryptosystem (encrypt, decrpyt)
and Gallagher for decrypt.
All users in a McEliece deployment share a set of common security parameters: n,k,t
There are some predefined steps: 
    1. Key generation;
    2. Message encryption;
    3. Message decryption.

After running this code, we obtained a figure (Figure_1.png), please see Docs section of this repository) which perfectly displays an overview 
of the methods' performance.

Note: This method is a post-quantum encryption candidate. A variant of this algorithm combined with NTS-KEM was entered into and selected
        during the third round of the NIST post-quantum encryption competition.


References: 
    1. McEliece, Robert J. (1978). "A Public-Key Cryptosystem Based on Algebraic Coding Theory" (PDF)
     DSN Progress Report. 44: 114-116 (see in Docs section)
    2. Classic McEliece Team (23 October 2022)  "Classic McEliece: conservative code-based cryptography: cryptosystem specification" 
    (PDF). Round 4 NIST Submission Overview. (https://classic.mceliece.org/ )
    3. http://www.infocobuild.com/education/audio-video-courses/electronics/ErrorCorrectingCodes-IISc-Bangalore/lecture-31.html 
    4. https://github.com/megabug/gallagher-research/blob/master/README.md 
'''
__author__ = "Alexandru Brezan", "Mircea Cristurean"
__version__ = 1.0
__project__ = "Encrypt and decrypt using McEliece and Gallagher methods"
__file__ = "matrixF.py"

import numpy as np
from random import shuffle, randrange
from time import time
import matplotlib.pyplot as plt


def generate_public_key(columns, rows):
    ''' The generate_public_key generates a random binary matrix for the public key

        :param int columns: defines number of columns for the generated public key
            :example: 34
        :param int rows: defines numer of row for the generated public key
            :example 34

        :return ndarray matrix: the matrix obtained for the public key
     '''
    w = 5
    # initializing with 0 (binary)
    matrix = np.zeros((rows, columns), dtype='bool')
    for index in range(0, rows):
        new_row = [True]*w+[False]*(columns-w)
        shuffle(new_row)
        matrix[index] = list(new_row)
    return matrix*1


def generate_private_key(G):
    ''' The generate_private_key generates a private key using the public key passed as parameter

        :param any G: its the matrix obtained from generate_public_key method
            :example: generate_private_key(generate_public_key(34, 34))

        :return any B_inv: the private key
     '''
    # Compute the rank of G
    rank_G = np.linalg.matrix_rank(G)
    rows, columns = G.shape
    # Select a subset of rows from G that form a basis for the row space of G
    # B = G[:rank_G, :]
    # print(G)
    R, order, LCS = rref(G)
    B = R[:, rows:]
    # Compute the inverse of B
    B_inv = np.linalg.inv(B)
    return B_inv


def rref(A):
    '''
        The rref method is an abreviation from Reduced Row Echelon Form. It is used to obtain the desired form
        of the generator matrix.

        :param any A: the matrix passed to be manipulated for obtaining Reduced Row Echelon Form
            :example generate_public_key(34, 34)

        :return A*1: the matrix obtained
        :return list jb
        :return list LCS
    '''
    m, n = A.shape
    i, j = 0, 0
    jb = []
    LCS = []
    while i < m and j < n:
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p*1 < 1:
            # The column is negligible, zero it out
            LCS.append(j)
            j = j + 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            for l in range(i+1, m):
                if A[l, j] == 1:
                    A[l, j:n] = np.logical_xor(A[l, j:n], A[i, j:n])
            i += 1
            j += 1
    for index in LCS:
        A = np.concatenate((A, np.vstack(A[:, index])), axis=1)
    for index in reversed(LCS):
        A = np.delete(A, index, 1)
    for i in range(m-1, 0, -1):
        L, = np.nonzero(A[0:i, i])
        for j in L:
            A[j, :] = np.logical_xor(A[j, :], A[i, :])
    return A*1, jb, LCS


def randomVector(length, w):
    '''
        This method generates a random vector used in McEliece method as error vector for encrypting

        :param int length: the length of the generated vector
            :example: 10
        :param int w
            :example: 10

        :return np.array(arr, dtype= 'bool'): the obatined vector
    '''
    # rand = randrange(0, length+1)
    arr = [True]*w+[False]*(length-w)
    shuffle(arr)
    return np.array(arr, dtype='bool')


def encrypt(message, G):
    '''
        The encrypt method, generates the ciphertext, using McEliece method of encryption

        :param any meesage: the message to be encrypted
            :example: 9865443211
        :param ndarray G: the matrix which is the public key used for encryption
            :example: generate_public_key(34, 34)

        :return ndarray ciphertext: the ecnrypted message using McEliece method, called ciphertext
    '''
    # Convert the message to a binary array
    message = np.array([int(x) for x in '{0:b}'.format(message)])
    print(message, "message")
    # Generate a random error vector
    e = np.random.randint(2, size=len(G[0]))
    e = randomVector(len(G[0]), 10)
    # Compute the ciphertext
    # ciphertext = message.dot(G) + e
    ciphertext = np.array(G@e, dtype='bool')
    return ciphertext*1


def decrypt(ciphertext, B_inv):
    '''
        The decrypt method is the McEliece method to decrypt the received message using the received ciphertext
        and the private key

        :param ndarray ciphertext: the encrypted messaged received
            :example: encrypt(9865443211,generate_public_key(34,34)) 
        :param ndarrat B_inv: the matrix used for decrypt (the private key)
            :example: generate_private_key(generate_public_key(34, 34))

        :return int message: the decrypted message
    '''
    # Compute the original message
    message = ciphertext.dot(B_inv) % 2
    # Convert the message to an integer
    message = int(''.join(map(str, message)), 2)
    return message


def decode_gallagher(ciphertext, B_inv, t):
    '''
        The decode_gallagher method is another decryption method which we decided to use

        :param ndarray ciphertext: the encrypted message received
            :example: encrypt(9865443211,generate_public_key(34,34)) 
        :param ndarray B_inv: the private key
            :example:  public_key*-1
        :param int t: the weight
            :example: 2

        :return np.array(ciphertext + e % 2, dtype='bool')*1 : the decrypted message using Gallager method
    '''
    n = len(B_inv[0])
    k = len(B_inv)
    # Compute the syndrome of the ciphertext
    # syndrome = ciphertext.dot(B_inv.T) % 2
    syndrome = np.array(ciphertext@B_inv.T % 2, dtype='bool')
    # Check if the syndrome is zero
    if not np.any(syndrome):
        return ciphertext
    # Initialize the error vector
    e = np.zeros(n)
    # Compute the weight of the syndrome
    weight = np.count_nonzero(syndrome)
    while weight > t:
        # Find the index of the least significant bit in the syndrome
        index = np.argmin(syndrome, 0)
        # Flip the corresponding bit in the error vector
        e[index] = 1
        # Update the syndrome
        # syndrome = (syndrome + B_inv[:, index]) % 2
        syndrome = np.array(syndrome @ B_inv[:, index] % 2)*1
        # Update the weight of the syndrome
        weight = np.count_nonzero(syndrome)
        print(syndrome)
    return np.array(ciphertext + e % 2, dtype='bool')*1


# Starting input:
n = 34
k = 34

# encrypt
public_key = generate_public_key(n, k)
ciphertext = encrypt(9865443211, public_key)
print(ciphertext, "cipher")

# decrypt:
# private_key = generate_private_key(public_key)
# plaintext = decrypt(ciphertext, private_key)
# print(plaintext)

# Example usage:
t = 2
plaintext = decode_gallagher(ciphertext, np.transpose(public_key), t)
plaintext = decode_gallagher(ciphertext, public_key*-1, t)
print(plaintext)

# t_values = [1, 2, 3, 4, 5, 6]
# times = []

# for t in t_values:
#     start = time()
#     plaintext = decode_gallagher(ciphertext, private_key, t)
#     end = time()
#     times.append(end - start)
#     print("t:", t, "time:", end - start)

# plt.plot(t_values, times)
# plt.xlabel('t values')
# plt.ylabel('Running time (seconds)')
# plt.title('Running time of decode_gallagher() function')
# plt.show()

print(__doc__)
