import numpy as np

def grid(Pmin, Pmax, M, R, T):
    def a(P):
        return (M*P*P/(365.25*365.25))**(1.0/3.0)

    def Tdur(P):
        return R/a(P)*(P/365.25)*0.5

    P = Pmin
    grids = []
    while P < Pmax:
        td = Tdur(P)
        grids.append((P, td))
        grids.append((P, td/2.0))
        grids.append((P, 2*td))

        dP = P*td/T/5.0

        P += dP

    return np.array(grids)
