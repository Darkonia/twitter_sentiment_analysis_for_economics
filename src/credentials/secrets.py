import base64
import pickle
from getpass import getpass

from cryptography.fernet import Fernet

from bld.project_paths import project_paths_join as ppj

# Authentication  process
try:
    key = getpass("Please enter password to decrypt credentials")
    key = key.encode("utf-8")
    key = base64.urlsafe_b64encode(key)
    cipher_suite = Fernet(key)
except ValueError:
    print("Invalid password")

# Use following code to encrypt your credentials
# pw = yourPassword
# pw = pw.encode('utf-8')
# pw = base64.urlsafe_b64encode(pw)
# cipher_suite = Fernet(pw)
# cipher_suite.decrypt(yourCredential.decode())    yourCredential as String

# encrypted credentials
# you will need to update with your own credentials to access Twitter API
credentials = {
    "consumer_key": b"gAAAAABeXmVqPMWttsCVRR0OHIqJZQ2c4ROZhnOwCsqUAflg-\
        CWM2pwYYXByoflL1DICTlIbkPgKXNbMWaPWe7JEjVDISdWRowp7SY0OFtftmRHMoq-pxAE=",
    "consumer_secret": b"gAAAAABeXmWUi9XtjT0QMQStN6uisTKRgH3CksJtGOf9LARbyBE-\
        DInIG5Uhp0Kc6AZBL7ts8qoNdQl0CCYzWFSQrgIhBgPk0AESCFAXjDWlrF2aq_-gh5dZny\
        DeMicRmLzxvjbv5Flh5TNC2IV9BaR2zUymnMk7oQ==",
    "access_token": b"gAAAAABeXmWu48weN-mWEQlIvWtLg7yVcIwZozJqcqosA7PdqhyaEmYOeW_\
        1cD_QzxKCzATSYNEGmYei5ZUEKrT14QRwkxk0FDhMR1n-3JY1ucipd_OlYOnUsMJSMDp9rdY-\
        dj3QsPkjVvBFjuR8mGhVh3qlWIrckA==",
    "access_token_secret": b"gAAAAABeXmXLEOO5-9HdFPoCO98PRgkOE9BjHzSIqYK7EmFVSa\
        Y7ttVoK444moJBXxTq0PndcPo1EBOp87tzWwXnL8WIJXcFE4mmcPRypShECNnluutNxENT6Q\
        jglM5TAD7hLbwz4RcD",
}
encrypted_credentials = credentials.copy()


def decrypt_credentials(cred):
    try:
        for c in cred.keys():
            cred[c] = cipher_suite.decrypt(cred[c]).decode()
        print("Credentials decrypted succesfully!")
    except Exception:
        print(
            "Invalid password. Please try again or update secrets.py with your own credentials."
        )
        print(
            "Ignore previous error message if your are not storing your credentials encrpyted"
        )
        cred = encrypted_credentials


if __name__ == "__main__":

    decrypt_credentials(credentials)

    with open(ppj("CREDENTIALS", "credentials.pickle"), "wb") as out_file:
        pickle.dump(credentials, out_file)
    print("asdf    " + ppj("CREDENTIALS", "credentials.pickle"))
