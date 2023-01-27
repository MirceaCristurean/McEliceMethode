import numpy as np
from random import shuffle, randrange
from time import time
import matplotlib.pyplot as plt

# Generate a random binary matrix for the public key
# def generate_public_key(n, k):
#     G = np.random.randint(2, size=(k, n))
#     return G
def generate_public_key(columns, rows):
    w = 5
    # initializare matrice cu zero binar
    matrix = np.zeros((rows, columns), dtype='bool')
    for index in range(0, rows):
        new_row = [True]*w+[False]*(columns-w)
        shuffle(new_row)
        matrix[index] = list(new_row)
    return matrix*1

# Generate the private key using the public key
def generate_private_key(G):
    # Compute the rank of G
    rank_G = np.linalg.matrix_rank(G)
    rows, columns = G.shape
    # Select a subset of rows from G that form a basis for the row space of G
    #B = G[:rank_G, :]
    #print(G)
    R, order, LCS = rref(G)
    B = R[:, rows:]
    # Compute the inverse of B
    B_inv = np.linalg.inv(B)
    return B_inv

# Calculate the row echelon form of the generator matrix
def rref(A):
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
    #rand = randrange(0, length+1)
    arr = [True]*w+[False]*(length-w)
    shuffle(arr)
    return np.array(arr, dtype='bool')

# Encrypt the message using the public key
def encrypt(message, G):
    # Convert the message to a binary array
    message = np.array([int(x) for x in '{0:b}'.format(message)])
    print(message, "message")
    # Generate a random error vector
    e = np.random.randint(2, size=len(G[0]))
    e = randomVector(len(G[0]), 10)
    # Compute the ciphertext
    #ciphertext = message.dot(G) + e
    ciphertext = np.array(G@e, dtype='bool')
    return ciphertext*1

# Decrypt the ciphertext using the private key
def decrypt(ciphertext, B_inv):
    # Compute the original message
    message = ciphertext.dot(B_inv) % 2
    # Convert the message to an integer
    message = int(''.join(map(str, message)), 2)
    return message


def decode_gallagher(ciphertext, B_inv, t):
    n = len(B_inv[0])
    k = len(B_inv)
    # Compute the syndrome of the ciphertext
    #syndrome = ciphertext.dot(B_inv.T) % 2
    syndrome = np.array(ciphertext@B_inv.T %2, dtype='bool')
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
        #syndrome = (syndrome + B_inv[:, index]) % 2
        syndrome = np.array(syndrome@ B_inv[:, index]%2)*1
        # Update the weight of the syndrome
        weight = np.count_nonzero(syndrome)
        print(syndrome)
    return np.array(ciphertext + e %2, dtype='bool')*1


# Starting input:
n = 34
k = 34

# encrypt
public_key = generate_public_key(n, k)
ciphertext = encrypt(9865443211, public_key)
print(ciphertext, "cipher")

# decrypt:
#private_key = generate_private_key(public_key)
#plaintext = decrypt(ciphertext, private_key)
#print(plaintext)

# Example usage:
t = 2
#plaintext = decode_gallagher(ciphertext, np.transpose(public_key), t)
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
