# Encrypt and decrypt using McEliece and Gallagher methods

## Authors:
    1. Alexandru Brezan
    2. Mircea Cristurean

 ### This file contains the implementation of encrypt and decrypt a message using 2 methods: McEliece cryptosystem (encrypt, decrpyt) and Gallagher for decrypt. 
 ### All users in a McEliece deployment share a set of common security parameters: n,k,t.
### There are some predefined steps: 
- Key generation;
- Message encryption;
- Message decryption.

### After running this code, we obtained a figure [Figure_1.png](../McEliceMethode/docs/Figure_1.png), please see Docs section of this repository) which perfectly displays an overview  of the methods' performance.

### Note: This method is a post-quantum encryption candidate. A variant of this algorithm combined with NTS-KEM was entered into and selected during the third round of the NIST post-quantum encryption competition.

# References: 
- McEliece, Robert J. (1978). "A Public-Key Cryptosystem Based on Algebraic Coding Theory" (PDF) DSN Progress Report. 44: 114-116 (please see:  [A_public_key_cryptosystem_McEliece.PDF](../McEliceMethode/docs/A_public_key_cryptosystem_McEliece.PDF))
- Classic McEliece Team (23 October 2022)  "Classic McEliece: conservative code-based cryptography: cryptosystem specification" (PDF). Round 4 NIST Submission Overview. [link](https://classic.mceliece.org/ )
- [http://www.infocobuild.com/education/audio-video-courses/electronics/ErrorCorrectingCodes-IISc-Bangalore/lecture-31.html](http://www.infocobuild.com/education/audio-video-courses/electronics/ErrorCorrectingCodes-IISc-Bangalore/lecture-31.html) 
- [https://github.com/megabug/gallagher-research/blob/master/README.md](https://github.com/megabug/gallagher-research/blob/master/README.md) 
