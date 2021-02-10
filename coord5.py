
import numpy as np

from Classes import Constants, Angle, Cart, KeplerOrbit



def ef2E(e, f, debug=False):

	f = Angle(f)

	E = np.arctan2(np.tan(f.rad/2),np.sqrt((1 + e)/(1 - e)))*2

	E = Angle(E)

	if debug:
		print("Eccentric Anomaly = {:.2f} rad".format(E.rad))

	return E

def eE2f(e, E, debug=False):

	E = Angle(E)

	f = np.arctan2(np.tan(E.rad/2), np.sqrt((1 - e)/(1 + e)))*2

	f = Angle(f)

	if debug:
		print("True Anomaly = {:.2f} rad".format(f.rad))

	return f

def orbit2State(a, e, f, mu):

	f = Angle(f)
	n = np.sqrt(mu/a**3)
	E = ef2E(e, f.rad)

	x = a*(np.cos(E.rad) - e)
	y = a*np.sqrt(1 - e**2)*np.sin(E.rad)
	z = 0

	v_x = -a*n*np.sin(E.rad)/(1 - e*np.cos(E.rad))
	v_y = a*np.sqrt(1 - e**2)*n*np.cos(E.rad)/(1 - e*np.cos(E.rad))
	v_z = 0

	r = Cart(x, y, z)
	v = Cart(v_x, v_y, v_z)

	return r, v

def rotateOrbitAngles(vector, w, i, O):

	vector = vector.rotate(-w, "z")
	vector = vector.rotate(-i, "x")
	vector = vector.rotate(-O, "z")

	return vector

def orbit2HeliocentricState(k_orbit, mu):

	r, v = orbit2State(k_orbit.a, k_orbit.e, k_orbit.f, mu)

	r = rotateOrbitAngles(r, k_orbit.w, k_orbit.i, k_orbit.O)
	v = rotateOrbitAngles(v, k_orbit.w, k_orbit.i, k_orbit.O)

	return r, v

if __name__ == "__main__":

	ef2E(0.5, 1.6, debug=True)
	eE2f(0.5, 1.07, debug=True)

	Mercury = KeplerOrbit()