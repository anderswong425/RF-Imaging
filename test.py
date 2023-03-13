import numpy as np
import time
from scipy.linalg import solve


def timing_decorator(num_runs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            total_time = 0
            for i in range(num_runs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                total_time += end - start
            avg_time = total_time / num_runs
            print(f"Avg execution time over {num_runs} runs: {avg_time:.5f} seconds")
            return result
        return wrapper
    return decorator


num_runs = 3


@timing_decorator(num_runs)
def xPRA(parameters, FrytB, FrytBat, Pinc, Ptot):

    Pryt = (Ptot-Pinc)/(20*np.log10(np.exp(1)))

    lambda_max = np.linalg.norm((FrytB.T @ Pryt), ord=2)

    Oimag = (np.linalg.solve(FrytBat + lambda_max * parameters['alpha'] * np.identity(FrytB.shape[1]), FrytB.T) @ Pryt)[parameters['pixel_size'][0]**2:]

    epr = 4*np.pi*(Oimag*0.5)/parameters['wavelength']

    epr[epr < 0] = 0

    epr = epr.reshape(parameters['pixel_size'], order='F')

    return epr


@timing_decorator(num_runs)
def xPRA_test(parameters, FrytB, FrytBat, Pinc, Ptot):

    Pryt = (Ptot-Pinc)/(20*np.log10(np.exp(1)))

    lambda_max = np.linalg.norm((FrytB.T @ Pryt), ord=2)

    Oimag = (np.linalg.solve(FrytBat + lambda_max * parameters['alpha'] * np.identity(FrytB.shape[1]), FrytB.T) @ Pryt)[parameters['pixel_size'][0]**2:]

    epr = 4*np.pi*(Oimag*0.5)/parameters['wavelength']

    epr[epr < 0] = 0

    epr = epr.reshape(parameters['pixel_size'], order='F')

    return epr


parameters = {}
parameters['alpha'] = 2
parameters['wavelength'] = 3e8/2.4e9
parameters['pixel_size'] = (60, 60)


FrytB = np.random.rand(380, 7200)
FrytBat = np.random.rand(7200, 7200)
Pinc = np.random.rand(380, 1)
Ptot = np.random.rand(380, 1)

A = xPRA(parameters, FrytB, FrytBat, Pinc, Ptot)

B = xPRA_test(parameters, FrytB, FrytBat, Pinc, Ptot)

print(np.allclose(A, B))
