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

        if deg:
            self.deg = ang%360
            self.rad = self.deg/180*np.pi
        else:
            self.rad = ang%(2*np.pi)
            self.deg = self.rad*180/np.pi

    def __str__(self):

        return "Angle: {:.2f}".format(self.deg)

    def __add__(self, other):

        return Angle(self.rad + other.rad)

    def __sub__(self, other):

        return Angle(self.rad - other.rad)

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

        self.lat = Angle(np.arctan2(EARTH_a**2*self.z, EARTH_b**2*self.h))
        self.lon = Angle(np.arctan2(self.y, self.x))

        rho = self.r/EARTH_a
        phiprime = Angle(np.arctan2(self.z, self.h))
        u = Angle(np.arctan2(EARTH_a*self.z, EARTH_b*self.h))

        self.alt = EARTH_a*(rho*np.cos(phiprime.rad) - np.cos(u.rad))/np.cos(self.lat.rad)

    def __str__(self):

        return "Cart Obj: x={:.2f} y={:.2f} z={:.2f}".format(self.x, self.y, self.z)

    def geoPos(self, pnt=True):
        """ returns the position in lat/lon/alt instead of x/y/z
        """

        if pnt:
            return "Cart Obj: lat={:.4f}N lon={:.4f}E alt={:.2f}km".format(self.lat.deg, self.lon.deg, self.alt/1000)
        else:
            return [self.lat.unmod(deg=True), self.lon.unmod(deg=True), self.alt]

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

    def __init__(a, e, i, O=None, w=None, f=None, w_tilde=None):
        pass
