
import numpy as np


class Constants:
    """ An object defining constants
    """ 

    def __init__(self):

        # Gravitational Constant
        self.G = 39.478 #AU^3 yr^-2 M_sun^-1

        # Astronomical Unit in meters
        self.AU = 1.496e+11 # m

        # Seconds in a year
        self.yr = 31557600 #s

        # Solar mass in kg
        self.M_sun = 1.989e30 #kg



class Vector3D:
    """ Basic function defining 3D cartesian vectors in [x, y, z] form

        Example:
            u = Vector3D(0, 1, 2)
            v = Vector3D(1, 1, 1)
            u is a vector [0, 1, 2]
            v is a vector [1, 1, 1]

    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.xyz = [x, y, z]

    def __add__(self, other):
        """ Adds two vectors as expected
        Example:
            w = u + v
            w is a vector [1, 2, 2]
        
        """
        result = Vector3D(self.x + other.x, \
                          self.y + other.y, \
                          self.z + other.z)
        return result

    def __sub__(self, other):
        """ Subtracts two vectors as expected
        Example:
            w = u - v
            w is a vector [-1, 0, 1]
        """
        result = Vector3D(self.x - other.x, \
                          self.y - other.y, \
                          self.z - other.z)
        return result


    def __mul__(self, other):
        """ Multipies a vector by a constant
        Example:
            w = u*3
            w is a vector [0, 3, 6]
        """
        result = Vector3D(self.x * other, \
                          self.y * other, \
                          self.z * other)
        return result

    def __str__(self):
        return '[ {:.4f}, {:.4f}, {:.4f}]'.format(self.x, self.y, self.z)

    def mag(self):
        """ Returns the geometric magnitude of the vector
        """
        result = (self.x**2 + self.y**2 + self.z**2)**0.5
        return result

    def dot(self, other):
        """ Returns the dot product of the vectors
        Example:
            w = u.dot(v)
            w = 3
        """ 
        result = self.x*other.x + self.y*other.y + self.z*other.z

        return result

    def cross(self, other):
        """ Returns the cross product of the vectors
        Example:
            w = u.cross(v)
            w is a vector [-1, 2, -1]
        """

        x = self.y*other.z - self.z*other.y
        y = self.z*other.x - self.x*other.z
        z = self.x*other.y - self.y*other.x

        result = Vector3D(x, y, z)
        return result



class Angle:
    """
    Angle object to easilt convert between radians and degrees

    Input:
    ang [float] - angle in radians
    deg [boolean] - if True, angle is given in degrees. If False (default), angle is given in radians
    Example:

        a = Angle(np.pi/2)
        or
        a = Angle(90, deg=True)
    """

    def __init__(self, ang, deg=False):

        if ang is not None:

            if deg:
                self.deg = ang%360
                self.rad = self.deg/180*np.pi
            else:
                self.rad = ang%(2*np.pi)
                self.deg = self.rad*180/np.pi

        else:

            self.deg = None
            self.rad = None

    def __str__(self):

        return "Angle: {:.2f}".format(self.deg)

    def __add__(self, other):

        return Angle(self.rad + other.rad)

    def __sub__(self, other):

        return Angle(self.rad - other.rad)

    def __neg__(self):

        return Angle(-self.rad)

    def unmod(self, deg=False):
        """ Retruns angle in range -180 < x < 180 instead of 0 < x < 360
        """

        if deg:
            if self.deg < 180:
                return self.deg
            return self.deg - 360
        else:
            if self.rad < np.pi:
                return self.rad
            return self.rad - 2*np.pi


    def isNone(self):

        if self.deg is None or self.rad is None:
            return True
        return False

class RightAsc:

    """ Quick object to convert an angle in degrees to a right ascension

    input: 
    angle [float] - angle in degrees
    Example:

        a = RightAsc(90)
        print(a)
        >> Right Ascension: 6.0h 0.0m 0.00s

    """
    def __init__(self, angle):

        self.angle = angle

        total_hours = angle/DEG_PER_HOUR

        self.hour, r = divmod(total_hours, 1)
        self.min, r = divmod(r*60, 1)
        self.sec = r*60


    def __str__(self):

        return "Right Ascension: {:}h {:}m {:.2f}s".format(self.hour, self.min, self.sec)

    def asFloat(self):
        # Returns the angle [deg] instead of right ascention
        
        return self.angle



class Cart:
    """
    A Cart(esian) object takes a geocentric vector defined as [x, y, z] [in meters], 
    and calculates various parameters from it
    """

    def __init__(self, x, y, z):

        try:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        except ValueError:
            print("[WARNING] Cartesian values must be a float")
            return None

        self.xyz = [self.x, self.y, self.z]
        self.h = (x*x + y*y)**0.5
        self.r = (x*x + y*y + z*z)**0.5

        self.phi =   Angle(np.arctan2(self.y, self.x))
        self.theta = Angle(np.arctan2(self.h, self.z))


    def __str__(self):

        return "Cart Obj: x={:.2f} y={:.2f} z={:.2f}".format(self.x, self.y, self.z)

    def rotate(self, ang, axis):
        """ Rotates vector <ang> degrees around an axis
            inputs:
            ang [Angle Obj] - angle to rotate coordinate system by
            axis ["x", "y", or "z"] - axis to rotate vector around 
        """
        
        if axis == "x":
            M = np.array([[1,          0,               0     ], \
                          [0, np.cos(ang.rad), np.sin(ang.rad)], \
                          [0, -np.sin(ang.rad), np.cos(ang.rad)]])
        elif axis == "y":
            M = np.array([[np.cos(ang.rad), 0, -np.sin(ang.rad)], \
                          [0, 1, 0], \
                          [np.sin(ang.rad), 0, np.cos(ang.rad)]])
        elif axis == "z":
            M = np.array([[np.cos(ang.rad), np.sin(ang.rad), 0], \
                          [-np.sin(ang.rad), np.cos(ang.rad), 0], \
                          [0, 0, 1]])
        else:
            print("Unrecognized Axis")
            return None

        vect = np.inner(M, self.xyz)

        return Cart(vect[0], vect[1], vect[2])


class KeplerOrbit:

    def __init__(self, a, e, i, O=None, w=None, f=None, w_tilde=None):

        self.a = a
        self.e = e
        self.i = Angle(i, deg=True)
        self.O = Angle(O, deg=True)
        self.w = Angle(w, deg=True)
        self.w_tilde = Angle(w_tilde, deg=True)

        if self.w.isNone() and not(self.O.isNone() or self.w_tilde.isNone()):
            self.w = self.w_tilde - self.O





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

def orbit2HeliocentricState(k_orbit, mu, f):

    r, v = orbit2State(k_orbit.a, k_orbit.e, f, mu)

    r = rotateOrbitAngles(r, k_orbit.w, k_orbit.i, k_orbit.O)
    v = rotateOrbitAngles(v, k_orbit.w, k_orbit.i, k_orbit.O)

    return r, v

if __name__ == "__main__":

    ef2E(0.5, 1.6, debug=True)
    eE2f(0.5, 1.07, debug=True)

    c = Constants()

    mu_sun = c.G*c.M_sun

    #(a, e, i, O=None, w=None, f=None, w_tilde=None)
    Mercury = KeplerOrbit(0.38709893, 0.20563069, 7.00487, O=48.33167, w_tilde=77.45645)
    Venus = KeplerOrbit(0.72333199, 0.00677323, 3.39471, O=76.68069, w_tilde=131.53298)
    Earth = KeplerOrbit(1.00000011, 0.01671022, 0.00005, O=-11.26064, w_tilde=102.94719)
    Mars = KeplerOrbit(1.52366231, 0.09341233, 1.85061, O=49.57854, w_tilde=336.04084)
    Jupiter = KeplerOrbit(5.20336301, 0.04839266, 1.30530, O=100.55615, w_tilde=14.75385)
    Saturn = KeplerOrbit(9.53707032, 0.05415060, 2.48446, O=113.71504, w_tilde=92.43194)
    Uranus = KeplerOrbit(19.19126393, 0.04716771, 0.76986, O=74.22988, w_tilde=170.96424)
    Neptune = KeplerOrbit(30.06896348, 0.00858587, 1.76917, O=131.72169, w_tilde=44.97135)
    Pluto = KeplerOrbit(39.48168677, 0.24880766, 17.14175, O=110.30347, w_tilde=224.06676)

    # https://ssd.jpl.nasa.gov/?sb_elem
    Ceres = KeplerOrbit(2.7653485, 0.07913825,  10.58682,  w=72.58981,  O=80.39320)

    Halley = KeplerOrbit(0.58597811, 0.96714291, 162.26269, w=111.33249, O=58.42008)

    planets = [Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Ceres, Halley]
    colours = ["#947876", "#bf7d26", "#479ef5", "#fa0707", "#c79e0a", "#bdba04", "#02edd6", "#2200ff", "#a3986c", "#030303", "#0aff78"]
    names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres", "Halley"]
    NO_OF_PLANETS = len(planets)
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as p3
    import matplotlib.animation as animation

    def make_planet(n, planet):
        data_x = []
        data_y = []
        data_z = []
        for f in np.linspace(0, 360, 1800):
            r, v = orbit2HeliocentricState(planet, mu_sun, f)
            data_x.append(r.x)
            data_y.append(r.y)
            data_z.append(r.z)

        data = np.array([data_x, data_y, data_z])
        return data


    def update(num, data, lines) :

        lines.set_data(data[0:2, num-1:num])
        lines.set_3d_properties(data[2,num-1:num])
        return lines

    def update_all(num, data, lines):

        l = [None]*NO_OF_PLANETS

        for i in range(NO_OF_PLANETS):
            l[i] = update(num, data[i][0], lines[i][0])

        return l

    # Attach 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    n = 100

    data = [None]*NO_OF_PLANETS
    lines = [None]*NO_OF_PLANETS

    for pp, p in enumerate(planets):
        data[pp] = [make_planet(n, p)]
        lines[pp] = [ax.plot(data[pp][0][0,0:1], data[pp][0][1,0:1], data[pp][0][2,0:1], \
                c=colours[pp], marker='o', label=names[pp])[0]]


    # Setthe axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-5.0, 5.0])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    ax.scatter([0], [0], [0], c="y", marker='o')

    for pp, planet in enumerate(planets):
        data_x = []
        data_y = []
        data_z = []
        for f in np.linspace(0, 360, 1800):
            r, v = orbit2HeliocentricState(planet, mu_sun, f)
            data_x.append(r.x)
            data_y.append(r.y)
            data_z.append(r.z)

        ax.plot(data_x, data_y, data_z, c=colours[pp])

    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_all, n, fargs=(data, lines),
                                  interval=50, blit=False)
    plt.legend()
    plt.show()