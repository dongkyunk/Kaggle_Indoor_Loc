from functools import wraps
import time
import logging 

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f'{func.__qualname__} starting')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info('{} complete in {} seconds\n'.format(
            func.__name__, int(end-start)))
        return result
    return wrapper