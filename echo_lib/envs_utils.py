#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List of utility functions for building environments

@author: thinh
"""

""" LIST OF COMPLETED FUNCTION
def build_empty_Arena(arenasize, cellsize, spacing = 2, K_wall = 1):
    ...
    arena_parameters = {"arenasize", "cellsize", "numcell"}
    return Arena, arena_parameters

def build_straight_wall(Arena, start_idx, length, direction, spacing=2):
    ...
    return newArena, end_idx

def set_target(Arena, location_idx, K_target=2):
    ...
    return Arena

def initialize_bat_fix_location(Arena, arena_params, start_idx, angular_direction):
    ...
    bat_location = {"location","angle","arrow","location_idx"}
    return activeArena, bat_location

def continuous_Arena(Arena, arena_params):
    ... convert gridworld matrix into a continuous space set of axis
    return Arena_xyk

def plot_Arena(Arena,arena_params, bat_location):
    ... plot arena on a continuous space with an arrow to represent a bat
    return fig, ax

def plot_view(fig, ax, bat_location, viewfield = [-60, +60], viewrange = 2):
    ... can only be called after calling plot_Arena
    ... The purpose is to plot out "view" of the bat using fan-out arrows
    return fig, ax, view_ax

def update_bat_location(bat_location, d_dist, d_angle, cellsize=0.1):
    ...first, move by distance d_dist. unit=meter. negative value --> move backward
    ...then, rotate by angle d_angle. unit=degrees. negative value --> turn left
    return new_location

def remove_objects_behind(inView, inView_dist, inView_angle, step=1):
    ... THIS IS A SUPPORT FUNCTION FOR get_objects_inView
    return nView, inView_dist, inView_angle

def get_objects_inView(Arena_xyk, bat_location,viewrange = 2):
    ...
    return inView, inView_dist, inView_angle


-------[things will be added in the future]-----------------------------------
def produce sonar output

def load_sonar_database


"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow

def build_empty_Arena(arenasize, cellsize, spacing = 2, K_wall = 1):
    numcell = int(arenasize/cellsize)
    Arena = np.zeros((numcell,numcell))
    if spacing >= 0:
        Arena[::(spacing + 1) , 0] = K_wall #adding West border wall
        Arena[0, ::(spacing + 1)] = K_wall #adding North border wall
        Arena[(int(np.ceil(spacing/2)))::(spacing + 1), numcell-1] = K_wall #adding East border wall
        Arena[numcell-1, (int(np.ceil(spacing/2)))::(spacing + 1)] = K_wall #adding East border wall
    arena_parameters = {"arenasize": arenasize,
                        "cellsize" : cellsize ,
                        "numcell"  : numcell
    }
    return Arena, arena_parameters

def build_straight_wall(Arena, start_idx, length, direction, spacing=2):
    newArena = np.empty(Arena.shape)
    if direction == 'N':
        end_idx = [start_idx[0] + length, start_idx[1]]
        Arena[(int(np.round(spacing/2))+start_idx[0]):end_idx[0]:(spacing+1), start_idx[1]] = 1
    elif direction == 'E':
        end_idx = [start_idx[0],start_idx[1] + length]
        Arena[start_idx[0],(int(np.round(spacing/2))+start_idx[1]):end_idx[1]:(spacing + 1)] = 1
    elif direction == 'S':
        end_idx = [start_idx[0] - length, start_idx[1]]
        Arena[(int(np.round(spacing/2))+end_idx[0]):start_idx[0]:(spacing+1), start_idx[1]] = 1
    elif direction == 'W':
        end_idx = [start_idx[0],start_idx[1] - length]
        Arena[start_idx[0], (int(np.round(spacing/2))+end_idx[1]):start_idx[1]:(spacing + 1)] = 1
    else:
        print("Directions only include {'N','E','S','W'}")
    newArena = np.copy(Arena)
    return newArena, end_idx

def set_target(Arena, location_idx, K_target=2):
    #remove previous target
    Arena[Arena == K_target] = 0
    Arena[location_idx] = K_target
    return Arena

def initialize_bat_fix_location(Arena, arena_params, start_idx, angular_direction):
    arenasize, cellsize, numcell = arena_params["arenasize"], arena_params["cellsize"], arena_params["numcell"]
    
    location = np.array([start_idx[1], start_idx[0]]).reshape(2,1) * cellsize
    activeArena = np.copy(Arena)
    activeArena[activeArena == -1] = 0
    if (activeArena[start_idx] == 0):
        activeArena[start_idx] = -1        
        arrow_dxdy = cellsize * np.array([np.cos(angular_direction * (np.pi/180)), np.sin(angular_direction * (np.pi/180))]).reshape(2,1)
    else:
        print("This is no joke! Get off the wall")
    
    bat_location = {"location" : location,
                    "angle" : angular_direction,
                    "arrow" : arrow_dxdy,
                    "location_idx" : start_idx}
    return activeArena, bat_location

def continuous_Arena(Arena, arena_params):
    arenasize, cellsize, numcell = arena_params["arenasize"], arena_params["cellsize"], arena_params["numcell"]
    x_axis = (np.ones((numcell,numcell)) * np.array([np.arange(numcell)])).reshape(numcell**2,1)
    y_axis = (np.ones((numcell,numcell)) * np.array([np.arange(numcell)]).T).reshape(numcell**2,1)
    A_unpack = Arena.reshape(numcell**2,1)
    
    x_axis = np.delete(x_axis, np.where(A_unpack==0)) * cellsize
    y_axis = np.delete(y_axis, np.where(A_unpack==0)) * cellsize
    k = np.delete(A_unpack, np.where(A_unpack==0))
    
    Arena_xyk = {"x_axis": x_axis,
                 "y_axis": y_axis,
                 "k"     : k}
    return Arena_xyk

def plot_Arena(Arena_xyk,arena_params,bat_location):
    x_axis, y_axis, k = Arena_xyk["x_axis"], Arena_xyk["y_axis"], Arena_xyk["k"]
    arenasize, cellsize, numcell = arena_params["arenasize"], arena_params["cellsize"], arena_params["numcell"]
    
    fig = plt.figure(figsize=(arenasize,arenasize))
    ax = fig.add_subplot(111)
    
    s1 = ((((fig.dpi)*arenasize) / numcell)/2)**2
    s2 = (1.5*((fig.dpi)*arenasize) / numcell)**2
    
    ax.scatter(x_axis[k==1],y_axis[k==1],s=s1,c='b',marker='o')
    ax.scatter(x_axis[k==2],y_axis[k==2],s=s2,c='g',marker='*')
    ax.grid()
    
    location, arrow= bat_location["location"], bat_location["arrow"]

    arrow_ax = ax.arrow(location[0,0],location[1,0],2*arrow[0,0],2*arrow[1,0], head_width=cellsize,head_length=cellsize*2,fc='r',ec='k',zorder = 2 )
    
    return fig, ax, arrow_ax

def get_preset_example(name='classic_v0-1'):
    if name == 'classic_v0-1':
        Arena, arena_params = build_empty_Arena(10,0.1,spacing=3)
        start_idx = (4,19)
        newArena, prev_end = build_straight_wall(Arena, start_idx, 80, 'N', spacing = 3)
        newArena, prev_end = build_straight_wall(newArena, prev_end, 60,'E',spacing=3)
        newArena, prev_end = build_straight_wall(newArena, prev_end, 30,'S',spacing=3)
        newArena, prev_end = build_straight_wall(newArena, prev_end, 30,'W',spacing=3)
        newArena, prev_end = build_straight_wall(newArena, prev_end, 30,'S',spacing=3)
        newArena = set_target(newArena, (66,66))
        activeArena, start_location = initialize_bat_fix_location(newArena, arena_params, (30,70), 45)
    if name == '3_poles_no_planter':
        newArena, arena_params = build_empty_Arena(10, 0.1, spacing=5)
        newArena[49,44] = 1
        newArena[49,54] = 1
        newArena[49,49] = 1
        activeArena, start_location = initialize_bat_fix_location(newArena, arena_params, (41,49), 90)
    if name == '2_poles_1_planter_0':
        newArena, arena_params = build_empty_Arena(10, 0.1, spacing=5)
        newArena[49, 44] = 2
        newArena[49, 54] = 1
        newArena[49, 49] = 1
        activeArena, start_location = initialize_bat_fix_location(newArena, arena_params, (41, 49), 90)
    if name == '2_poles_1_planter_1':
        newArena, arena_params = build_empty_Arena(10, 0.1, spacing=5)
        newArena[49, 44] = 1
        newArena[49, 54] = 2
        newArena[49, 49] = 1
        activeArena, start_location = initialize_bat_fix_location(newArena, arena_params, (41, 49), 90)
    if name == '2_poles_1_planter_2':
        newArena, arena_params = build_empty_Arena(10, 0.1, spacing=5)
        newArena[49, 44] = 1
        newArena[49, 54] = 1
        newArena[49, 49] = 2
        activeArena, start_location = initialize_bat_fix_location(newArena, arena_params, (41, 49), 90)
    return newArena, activeArena, arena_params, start_location


def plot_view(fig, ax, bat_location, viewfield = [-45, +45], viewrange = 2.5):
    location, angle = bat_location["location"], bat_location["angle"]
    dxdy = np.zeros(bat_location["arrow"].shape)
    for i in np.arange(viewfield[0],viewfield[1]+1,5):
        #calculate dxdy
        temp_angle = angle + i
        dxdy = viewrange * np.array([np.cos(temp_angle * (np.pi/180)), np.sin(temp_angle * (np.pi/180))]).reshape(2,1)
        view_ax = ax.arrow(location[0,0],location[1,0],dxdy[0,0],dxdy[1,0],fc = 'y', ec='y',zorder=1)
    return fig, ax, view_ax


def update_bat_location(bat_location, d_dist, d_angle, cellsize=0.1): # move d_dist then turn d_angle , neg angle to go left
    location, angle = bat_location["location"], bat_location["angle"]
    
    dxdy = d_dist * np.array([np.cos(angle * (np.pi/180)), np.sin(angle * (np.pi/180))]).reshape(2,1)
    location = location + dxdy
    angle = angle + d_angle
    
    arrow = cellsize * np.array([np.cos(angle * (np.pi/180)), np.sin(angle* (np.pi/180))]).reshape(2,1)
    location_idx = tuple(np.round(location/0.1).squeeze().astype('int'))
    
    new_location = {"location" : location,
                    "angle"    : angle,
                    "arrow"    : arrow,
                    "location_idx" : location_idx}
    return new_location


def remove_objects_behind(inView, inView_dist, inView_angle, step=1):
    #find a pair/group of vector that's close to each other, but how close?
    angle_norm = np.round( inView_angle * (1/step) )
    del_cache=[]
    for i, a in enumerate(angle_norm):
        if inView_dist[i,0] != np.amin(inView_dist[angle_norm == a]):
            del_cache.append(i)
    inView = np.delete(inView, del_cache, axis=0)
    inView_dist = np.delete(inView_dist,del_cache, axis=0)
    inView_angle = np.delete(inView_angle,del_cache, axis=0)
    return inView, inView_dist, inView_angle


def get_objects_inView(Arena_xyk, bat_location,viewfield=45, viewrange = 2.5):
    step = 5 # number of degree the angle vector will be normalized to --> remove objects behind
    inSpace = np.array([Arena_xyk["x_axis"],Arena_xyk["y_axis"],Arena_xyk["k"]]).reshape(3,Arena_xyk["k"].shape[0]).T
    location, angle, arrow = bat_location["location"], bat_location["angle"], bat_location["arrow"]
    
    winx_lim = np.array([ location[0]-viewrange , location[0]+viewrange ]).squeeze()
    winy_lim = np.array([ location[1]-viewrange , location[1]+viewrange ]).squeeze()
    #windowing size 2*viewingrange.
    temp_x = inSpace[:,0]
    temp_y = inSpace[:,1]
    inView = np.delete(inSpace, np.where( (temp_x < winx_lim[0]) | (temp_x > winx_lim[1]) | (temp_y < winy_lim[0]) | (temp_y > winy_lim[1]) ), axis = 0)
    
    # calculate distance
    inView_dist = np.linalg.norm(inView[:,0:2] - location.reshape(1,2) , axis=1).reshape(inView.shape[0],1)
    inView = np.delete(inView, np.where(inView_dist.squeeze() > viewrange ),axis = 0)
    inView_dist = np.delete(inView_dist, np.where(inView_dist.squeeze() > viewrange ),axis = 0)
    #calculate angle
    u = arrow.reshape(2,1)
    v = inView[:,0:2] - location.reshape(1,2)

    inView_angle = np.degrees( -np.arctan2(v[:,1],v[:,0]) + np.arctan2(u[1],u[0]) ).reshape(inView.shape[0],1)
    inView_angle[inView_angle > 180] = inView_angle[inView_angle>180] -  360
    inView_angle[inView_angle <-180] = inView_angle[inView_angle<-180] + 360
    inView = np.delete(inView, np.where((inView_angle.squeeze()>viewfield) | (inView_angle.squeeze()<-viewfield) ), axis=0)
    inView_dist = np.delete(inView_dist, np.where((inView_angle.squeeze()>viewfield) | (inView_angle.squeeze()<-viewfield) ), axis=0)
    inView_angle = np.delete(inView_angle, np.where((inView_angle.squeeze()>viewfield) | (inView_angle.squeeze()<-viewfield) ), axis=0)
    inView_angle = -1 * inView_angle

    inView, inView_dist, inView_angle = remove_objects_behind(inView, inView_dist, inView_angle, step=step)
    
    return inView, inView_dist, inView_angle

