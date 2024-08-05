import numpy as np
import subprocess as sp
from scipy import integrate as scp
from scipy import interpolate as interp
import matplotlib.pyplot as plt

def IntegFromData(xvalues,yvalues,sup):
    if (sup > np.max(xvalues)):
        return print("Extremo superior no válido, intente con otro valor.")
    else:
        function = interp.CubicSpline(xvalues,yvalues)
        result = scp.quadrature(function, np.min(xvalues), sup, tol=0.000001 )
        # result is a tuple (integral_value,error)
        return result[0]

sp_input = ["1", "0", "1", "0.0", "-0.0"]

energies = []
localization = []

extr = 1.2

for item in np.arange(0.001,0.501,0.001):
    sp_input[3] = str(item)

    with open("Read/sp.dat","w") as data:
        data.write(" \t".join(sp_input))

    # Correr el código
    sp.run(["./cwf"])

    wf = np.loadtxt('Out/wf.dat', usecols = (0,1))
    R_N = np.loadtxt('Out/bigr.dat', max_rows = 1)

    energies.append(item)
    localization.append( IntegFromData( wf[:,0], wf[:,1]**2, extr*R_N ) )

with open("local.dat","w") as local:
    for f, b in zip(energies, localization):
        local.write(" {0:.3f} \t {1:5f} \n".format(f, b))

plt.xlabel("Energy [MeV]")
plt.ylabel("L(E)")

plt.legend()
plt.plot(energies,localization,"-")

plt.savefig("localization.pdf",format="pdf")
plt.show()