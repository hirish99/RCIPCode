import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps


n = 16
lege_nodes, lege_weights, _ = sps.legendre(n).weights.T
def make_panels(panel_boundaries):
    panel_lengths = np.diff(panel_boundaries)
    panel_centers = 0.5*(panel_boundaries[1:] + panel_boundaries[:-1])

    return (
            panel_lengths.reshape(-1, 1) * 0.5 * lege_nodes
            + panel_centers.reshape(-1, 1))

def get_coordinates_boundary_ellipse(t):
    stretch=3
    retMe = np.array([
    stretch*np.cos(2*np.pi*t),
    np.sin(2*np.pi*t),
    ])
    return retMe.reshape(2,-1)

npan = 20
np.savetxt('npan.np', np.array([0, int(npan)]))
npoin = npan*16
panel_boundaries = np.linspace(0, 1, npan+1)
x,y = get_coordinates_boundary_ellipse(make_panels(panel_boundaries))
point_charges = np.array([np.array([-2,2])])
target = np.array([1,0])


plt.scatter(x,y)
plt.scatter(x[:100],y[:100])
plt.scatter(point_charges.T[0],point_charges.T[1],marker='x', label="Source")
plt.scatter(target[0],target[1],c='r', label="Target")
plt.axis('equal')
plt.legend()
plt.show()


def G(x,y):
    return (-1/(2*np.pi))*np.log(np.linalg.norm(x-y,2))
points = np.empty((2,len(x)))
points[0,:] = x
points[1,:] = y

potential = np.zeros(len(x))
for i in range(len(points.T)):
    for j in range(len(point_charges)):
        potential[i]+=G(points.T[i],point_charges[j])

def get_potential(point):
    potential = 0
    for j in range(len(point_charges)):
        potential+=G(point,point_charges[j])
    return potential

xline = x
yline = y
zline = potential
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xline, yline, zline, 'green')
ax.plot3D(xline, yline, np.zeros(len(zline)), 'grey')
plt.title("Potential on Boundary")
plt.show()

print(get_potential(target))

np.savetxt("bc_potential.np", potential)