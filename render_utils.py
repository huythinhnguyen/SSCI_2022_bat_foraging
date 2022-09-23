from matplotlib import pyplot as plt

import numpy as np

def plot_step(obj, bat, echo, with_echo=False):
    arenasize, cellsize, numcell = (10, 0.1, 100)
    viewfield = [-45,+45]
    viewrange = 2.5
    if with_echo:
        fig,(ax1, ax2) = plt.subplots(1,2,figsize=(2.5*arenasize,arenasize), gridspec_kw={'width_ratios': [2,3]})
    else:
        fig, ax1 = plt.subplots(1,1, figsize=(arenasize, arenasize))
    # plot the matrix
    x_axis, y_axis, k = (obj._coordinates[:,0], obj._coordinates[:,1], obj._coordinates[:,2])

    s1 = ((((fig.dpi)*arenasize) / numcell))**2
    s2 = (2*((fig.dpi)*arenasize) / numcell)**2
    
    ax1.scatter(x_axis[k==1], y_axis[k==1], s=s1, c='b', marker='o')
    ax1.scatter(x_axis[k==2], y_axis[k==2], s=s2, c='g', marker='*')
    ax1.grid()
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_xlim(-9,9)
    ax1.set_ylim(-9,9)
    location, angle = (bat._tracker[0,:2], bat._tracker[0,2])
    dxdy = cellsize * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    ax1.arrow(location[0], location[1], 2*dxdy[0], 2*dxdy[1],
                         head_width=cellsize, head_length=cellsize*2, fc='r', ec='k', zorder=2)
    ray = np.zeros(2)

    for i in np.arange(viewfield[0], viewfield[1]+1, 5):
        temp_angle = angle+i
        ray = viewrange * np.array([np.cos(np.radians(temp_angle)), np.sin(np.radians(temp_angle))])
        ax1.arrow(location[0], location[1], ray[0], ray[1], fc='y', ec='y', zorder=1)
    if with_echo:
        ax2.bar(range(50), echo._echo[:50], alpha=0.7, label='left')
        ax2.bar(range(50), echo._echo[50:], alpha=0.7, label='right')
        ax2.legend()
        dist_z = np.floor(100*np.linspace(0.04,3.51,50))/100
        ax2.set_xticks(np.arange(len(dist_z))[::4])
        ax2.set_xticklabels(np.round(dist_z,2)[::4])
        ax2.set_title('Compressed envelope of current time step')
        ax2.set_xlabel('Distance (meter)')
        ax2.set_ylabel('Normalized (a.u)')
    else:
        ax2 = None
    return fig, ax1, ax2


def render_trajectory(obj, bat, echo, act, status, with_echo=False):
    fig, ax1, ax2 = plot_step(obj, bat, echo, with_echo=with_echo)
    history_x, history_y = (act.history['bat'][:,0], act.history['bat'][:,1])
    hist_x_0 = history_x[np.where(act.history['loco'][:,0]==0)]
    hist_x_1 = history_x[np.where(act.history['loco'][:,0]==1)]
    hist_y_0 = history_y[np.where(act.history['loco'][:,0]==0)]
    hist_y_1 = history_y[np.where(act.history['loco'][:,0]==1)]
    if len(hist_x_0)>0:
        ax1.scatter(hist_x_0, hist_y_0, s=1.5, c='dodgerblue', alpha=0.8)
        
    if len(hist_x_1)>0:
        ax1.scatter(hist_x_1, hist_y_1, s=1.5, c='red', alpha=0.8)

    if status.hit!=0:
        marker_spec = 'rx' if status.hit==2 else 'gx'
        ax1.plot(bat._tracker[0,0], bat._tracker[0,1], marker_spec, markeredgewidth=4, alpha=0.5, markersize=14)
    return fig, ax1, ax2

