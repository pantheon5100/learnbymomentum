import random  
import string  
import secrets

def get_version(length=10):  

    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))  

  
