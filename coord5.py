
import numpy as np

from Classes import Constants, Angle



def ef2E(e, f, debug=False):

	f = Angle(f)

	E = np.arctan2(np.tan(f.rad/2),np.sqrt((1 + e)/(1 - e)))*2

	E = Angle(E).rad

	if debug:
		print("Eccentric Anomaly = {:.2f} rad".format(E))

	return E

def eE2f(e, E, debug=False):

	E = Angle(E)

	f = np.arctan2(np.tan(E.rad/2), np.sqrt((1 - e)/(1 + e)))*2

	f = Angle(f).rad

	if debug:
		print("True Anomaly = {:.2f} rad".format(f))

	return f

if __name__ == "__main__":

	ef2E(0.5, 1.6, debug=True)
	eE2f(0.5, 1.07, debug=True)