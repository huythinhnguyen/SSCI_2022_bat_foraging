import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg.linalg import multi_dot

try:
    from echo_lib.envs_utils import update_bat_location, get_objects_inView
    from echo_lib.echo_transform_utils import get_total_envelope_z
except ImportError as e:
    print('check whether functions name has been altered'.format(e))

# helper functions:
def cartesian2polar(xy):
    if xy.shape[1] != 2:
        raise ValueError('input for `catersian2polar()` need to be in shape of (n,2)')
    else:
        x, y = (xy[:,0], xy[:,1])
    r = np.sqrt( np.power(x, 2) + np.power(y, 2))
    theta = np.degrees(np.arctan2(y, x))
    polar = np.zeros(xy.shape)
    polar[:,0] = r
    polar[:,1] = theta
    return polar


def polar2cartesian(polar):
    xy = np.zeros(polar.shape)
    if polar.shape[1] != 2:
        raise ValueError('input for `polar2catersian()` need to be in shape of (n,2)')
    else:
        r, theta = (polar[:,0], polar[:,1])
    x = r * np.cos( np.radians(theta) )
    y = r * np.sin( np.radians(theta))
    xy[:,0] = x
    xy[:,1] = y
    return xy


def xy_bat_reference(xy, bat_local):
    xy_bat = np.zeros(xy.shape)
    xy_bat[:,0] = xy[:,0] - bat_local[:,0]
    xy_bat[:,1] = xy[:,1] - bat_local[:,1]
    return xy_bat


def polar_bat_reference(obj_coordinates, bat_tracker):
    bat_local = bat_tracker[:,:2]
    obj_polar = obj_coordinates[:,:2]
    obj_polar = xy_bat_reference(obj_polar, bat_local)
    obj_polar = cartesian2polar(obj_polar)
    return obj_polar


class ObjCoordinates:
    def initialize_preset(self, preset):

        if preset >=0:
            coordinates = np.array([]).reshape(0,3)
            food = np.array([0, 0, 1], dtype=np.float64).reshape(1,3)
            coordinates = np.vstack((coordinates, food))
            # inner wall, NS:
            y_coor = np.linspace(-4, 4, 21).reshape(-1,1)
            x_coor = 4*np.ones(21).reshape(-1,1)
            k_coor = 2*np.ones(21).reshape(-1,1)
            temp = np.hstack(( x_coor, y_coor, k_coor))
            coordinates = np.vstack(( coordinates, temp ))
            temp = np.hstack(( (-1)*x_coor, y_coor, k_coor ))
            coordinates = np.vstack(( coordinates, temp ))

            # inner wall, WE:
            x_coor = np.linspace(-4, 4, 21).reshape(-1,1)
            y_coor = 4*np.ones(21).reshape(-1,1)
            k_coor = 2*np.ones(21).reshape(-1,1)
            temp = np.hstack(( x_coor, y_coor, k_coor))
            coordinates = np.vstack(( coordinates, temp ))
            temp = np.hstack(( x_coor, (-1)*y_coor, k_coor ))
            coordinates = np.vstack(( coordinates, temp ))
            
            # inner wall, NS:
            y_coor = np.linspace(-7, 7, 36).reshape(-1,1)
            x_coor = 7*np.ones(36).reshape(-1,1)
            k_coor = 2*np.ones(36).reshape(-1,1)
            temp = np.hstack(( x_coor, y_coor, k_coor))
            coordinates = np.vstack(( coordinates, temp ))
            temp = np.hstack(( (-1)*x_coor, y_coor, k_coor ))
            coordinates = np.vstack(( coordinates, temp ))

            # inner wall, WE:
            x_coor = np.linspace(-7, 7, 36).reshape(-1,1)
            y_coor = 7*np.ones(36).reshape(-1,1)
            k_coor = 2*np.ones(36).reshape(-1,1)
            temp = np.hstack(( x_coor, y_coor, k_coor))
            coordinates = np.vstack(( coordinates, temp ))
            temp = np.hstack(( x_coor, (-1)*y_coor, k_coor ))
            coordinates = np.vstack(( coordinates, temp ))

        if preset <0:
            coordinates = np.array([]).reshape(0,3)
            food = np.array([0, 0, 1], dtype=np.float64).reshape(1,3)
            coordinates = np.vstack((coordinates, food))
            polar1 = 5 * np.ones((75, 1))
            temp = np.linspace(-180, 180, 76)[:75].reshape((75,1))
            polar1 = np.hstack(( polar1, temp ))
            polar2 = 8 * np.ones((125, 1))
            temp = np.linspace(-180, 180, 126)[:125].reshape((125,1))
            polar2 = np.hstack((polar2, temp))

            polar = np.vstack((polar1, polar2))
            xy = polar2cartesian((polar))
            xyk = np.hstack((xy, 2*np.ones((125+75, 1))))
            coordinates = np.vstack(( coordinates, xyk ))
        return coordinates


    def get_food_list(self, preset=1):
        if preset >0:
            AA = np.random.rand(520,2)
            AA[:,0] = 12*AA[:,0] - 6
            AA[:,1] = AA[:,1] + 5
            BB = np.random.rand(520,2)
            BB[:,0] = 12*BB[:,0] - 6
            BB[:,1] = -1*BB[:,1] - 5
            CC = np.random.rand(400,2)
            CC[:,0] = CC[:,0] + 5
            CC[:,1] = 12*CC[:,1] - 6
            DD = np.random.rand(400,2)
            DD[:,0] = -1*DD[:,0] - 5
            DD[:,1] = 12*DD[:,1] - 6
            xy = np.vstack((AA, BB, CC, DD))
            polar = cartesian2polar(xy)
            xy = np.hstack(( xy, np.ones((xy.shape[0],1)) ))
            polar = np.hstack(( polar, np.ones((polar.shape[0], 1)) ))
        if preset <0:
            polar = np.random.rand(2000, 2)
            polar[:,0] = polar[:,0] + 6
            polar[:,1] = 360*polar[:,1] - 180
            xy = polar2cartesian(polar)
            polar = np.hstack((polar, np.ones((polar.shape[0], 1))))
            xy = np.hstack((xy, np.ones((xy.shape[0],1))))
        if preset == -2 or preset == 2:
            polar[:,2] = 2
            xy[:,2] = 2
            
        return xy, polar


    def __init__(self, coordinates=None, preset=1, pole_radius=0.4, plant_radius=0.4):
        self.preset = preset
        self.radius = {'pole': pole_radius, 'plant': plant_radius}
        if coordinates==None: coordinates = self.initialize_preset(self.preset)
        self._coordinates = coordinates
        self.FOOD_XY_LS, self.FOOD_POLAR_LS = self.get_food_list(self.preset)
        self.FOOD_FROM_ANGLES  = [0, 90, -180, -90]
        self.FOOD_TO_ANGLES = [30, 120, -150, -60]

        if preset==100: self._coordinates[0] = np.asarray([6., 1., 1.])
        

    def get_random_food(self, area_index, xy_ls=None, polar_ls=None):
        i = area_index % len(self.FOOD_FROM_ANGLES)
        from_angle = self.FOOD_FROM_ANGLES[i]
        to_angle = self.FOOD_TO_ANGLES[i]
        xy_ls = self.FOOD_XY_LS if xy_ls == None else xy_ls
        polar_ls = self.FOOD_POLAR_LS if polar_ls == None else polar_ls
        food_list = xy_ls[np.where((polar_ls[:,1]>from_angle)*(polar_ls[:,1]<to_angle))]
        n = food_list.shape[0]
        idx = np.random.randint(n)
        selected = food_list[idx]
        return selected


    def set_new_food(self, index):
        if self.preset < 10:
            self._coordinates[self._coordinates[:,2]==1] = self.get_random_food(index)
        if self.preset > 10 and self.preset < 100:
            from_to_ls = [(0,5), (5,10)]
            from_angle, to_angle = from_to_ls[self.preset - 11]
            xy_ls = self.FOOD_XY_LS
            polar_ls = self.FOOD_POLAR_LS
            food_list = xy_ls[np.where((polar_ls[:,1]>from_angle)*(polar_ls[:,1]<to_angle))]
            n = food_list.shape[0]
            idx = np.random.randint(n)
            self._coordinates[self._coordinates[:,2]==1] = food_list[idx]
        if self.preset>100: self._coordinates[self._coordinates[:,2]==1] = np.asarray([-5.5, 0., 1.])
        return self._coordinates[self._coordinates[:,2]==1]


    def reset(self):
        self._coordinates = self.initialize_preset(self.preset)
        if self.preset==100: 
            self._coordinates[0] = np.asarray([6., 1., 1.])
            return self._coordinates
        self.set_new_food(0)
        return self._coordinates


class BatPosition:
    def init_preset(self, preset, xy_jitter=0.1, angle_jitter=10):
        if preset == 1 or preset == 2:
            #tracker = np.array([5.5, -2.5, 90.0], dtype=np.float64).reshape(1,3)
            tracker = np.array([0.0, -5.5, 0.0], dtype=np.float64).reshape(1,3)
        if preset > 10:
            tracker = np.array([5.5, -1.5, 90.0], dtype=np.float64).reshape(1,3)
        if preset ==-1 or preset==-2:
            tracker = np.array([0.0, -6.5, 0.0], dtype=np.float64).reshape(1,3)
        jit = np.array([xy_jitter, xy_jitter, angle_jitter])
        jittering = (np.multiply(2*jit, np.random.rand(3)) - jit).reshape(1,3)
        tracker = tracker + jittering
        if preset==100: tracker = np.array([0.0, -5.5, 0.0], dtype=np.float64).reshape(1,3)
        return tracker


    def __init__(self, preset=1):
        self.preset = preset
        self._tracker = self.init_preset(self.preset)
        self.last_move = 0
        self.last_turn = 0

        
    def reset(self):
        del self._tracker
        self._tracker = self.init_preset(self.preset)
        return self._tracker


    def longitudinal_move(self, distance):
        current_local = {'location': self._tracker[:,:2].reshape(2,1), 'angle': self._tracker[:,2]}
        next_local = update_bat_location(current_local, distance, 0)
        self._tracker[:,:2] = next_local['location'].reshape(1,2)
        self.last_move = distance
        return self._tracker


    def rotational_move(self,angle):
        current_local = {'location': self._tracker[:,:2].reshape(2,1), 'angle': self._tracker[:,2]}
        next_local = update_bat_location(current_local, 0, angle)
        a = next_local['angle']
        a = (a+360) if a<-180 else (a-360) if a>180 else a
        self._tracker[:,2] = a
        self.last_turn = angle

        
class BatEcho:
    def __init__(self):
        self._echo = np.zeros(100)
        self.glomax = 5.73


    def call(self, obj, bat, trunc=50):
        local = {'location': bat._tracker[:,:2].reshape(2,1), 'angle': bat._tracker[:,2],
                 'arrow': np.array([np.cos(np.radians(bat._tracker[:,2])), np.sin(np.radians(bat._tracker[:,2]))], dtype=np.float32)}
        xyk = {'x_axis': obj._coordinates[:,0], 'y_axis': obj._coordinates[:,1], 'k': obj._coordinates[:,2]}
        inView, inView_dist, inView_angle = get_objects_inView(xyk, local)
        data = get_total_envelope_z(inView, inView_dist, inView_angle)
        observation = np.concatenate((data['left'][:trunc],data['right'][:trunc])).reshape(100,)
        self._echo = observation/self.glomax
        return self._echo
    

    def reset(self):
        self._echo = np.zeros(self._echo.shape)
        return self._echo


class BatAction:
    def __init__(self, decode_dict):
        self.decode = decode_dict
        self.num_actions = len(self.decode)
        self.index = 0
        self.onehot = np.zeros(self.num_actions)
        self.onehot[0] = 1
        self.history = {'bat': np.array([]).reshape(0,3), 'loco': np.array([]).reshape(0,3)}
        self.steps = 0


    def as_onehot(self, index):
        onehot = np.zeros(self.onehot.shape)
        onehot[index] = 1
        return onehot


    def update(self, obj, bat, echo, status, action_id=None, action_txt=None, out='onehot'):
        if action_id!=None:
            self.index = action_id.reshape(1,)[0]
            self.onehot = self.as_onehot(self.index)
        elif action_txt!=None:
            self.index = self.decode[action_txt]
            self.onehot = self.as_onehot(self.index)
        else:
            raise ValueError('must at least input action as id or text.\n' + str(self.decode))

        status.check(obj, bat)
        echo.call(obj,bat)

        loco_temp = np.array([self.index, bat.last_move, bat.last_turn])
        bat_temp = bat._tracker.reshape(3,)
        self.history['bat'] = np.vstack(( self.history['bat'], bat_temp ))
        self.history['loco']= np.vstack((self.history['loco'], loco_temp))

        self.steps += 1

        if out=='onehot': 
            return self.onehot
        else:
            return self.index


    def reset(self):
        self.history = {'bat': np.array([]).reshape(0,3), 'loco': np.array([]).reshape(0,3)}
        self.index = 0
        self.onehot = self.as_onehot(self.index)
        self.steps = 0
        return None



class BatStatus:
    def __init__(self, diet='pole'):
        self.hit = 0
        self.food_distance = None
        self.food_azimuth = None


    def reset(self):
        self.hit = 0
        self.food_distance = None
        self.food_azimuth = None
        return None


    def hit_an_object(self, obj, bat):
        objects = polar_bat_reference(obj._coordinates, bat._tracker)
        closest_arg = np.argmin(objects[:,0].reshape(-1,))
        k = obj._coordinates[closest_arg,2]
        hitzone = obj.radius['pole'] if k==1 else obj.radius['plant'] if k==2 else 0
        self.hit = k if objects[closest_arg,0] < hitzone else 0
        return self.hit

    
    def check(self, obj, bat):
        self.hit_an_object(obj, bat)
        if len(obj._coordinates[obj._coordinates[:,2]==1])>0:
            food = obj._coordinates[obj._coordinates[:,2]==1][0].reshape(1,3)
        else:
            food = np.array([0,0,1]).reshape(1,3)
        polar = polar_bat_reference(food, bat._tracker).reshape(2,)
        self.food_distance = polar[0]
        azi_temp = bat._tracker[0,2] - polar[1]
        if azi_temp >= 180:
            self.food_azimuth = azi_temp - 360
        elif azi_temp <-180:
            self.food_azimuth = azi_temp + 360
        else: self.food_azimuth = np.copy(azi_temp)
        return None
