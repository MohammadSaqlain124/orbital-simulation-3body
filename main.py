import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Constants
G = 6.67430e-11
AU = 1.496e11


# Masses
M_sun = 1.989e30
M_earth = 5.972e24
M_jupiter = 1.898e27


# Initial Positions (meters)
r_sun = np.array([0.0, 0.0])
r_earth = np.array([1.0 * AU, 0.0])
r_jupiter = np.array([5.2 * AU, 0.0])


# Initial Velocities (m/s)
v_sun = np.array([0.0, 0.0])
v_earth = np.array([0.0, 29780.0])
v_jupiter = np.array([0.0, 13070.0])


# Time Setup
dt = 60 * 60 * 48   # 12 hours
steps = 5000


# Storage
earth_positions = []
sun_positions = []
jupiter_positions = []
energies = []


# Acceleration Function
def acceleration(r1, r2, m2):
    r = r2 - r1
    dist = np.linalg.norm(r)
    return G * m2 * r / dist**3

# Initial accelerations
a_sun = acceleration(r_sun, r_earth, M_earth) + acceleration(r_sun, r_jupiter, M_jupiter)
a_earth = acceleration(r_earth, r_sun, M_sun) + acceleration(r_earth, r_jupiter, M_jupiter)
a_jupiter = acceleration(r_jupiter, r_sun, M_sun) + acceleration(r_jupiter, r_earth, M_earth)


# Simulation (Verlet)
for _ in range(steps):

    # Update positions
    r_sun += v_sun * dt + 0.5 * a_sun * dt**2
    r_earth += v_earth * dt + 0.5 * a_earth * dt**2
    r_jupiter += v_jupiter * dt + 0.5 * a_jupiter * dt**2

    # New accelerations
    new_a_sun = acceleration(r_sun, r_earth, M_earth) + acceleration(r_sun, r_jupiter, M_jupiter)
    new_a_earth = acceleration(r_earth, r_sun, M_sun) + acceleration(r_earth, r_jupiter, M_jupiter)
    new_a_jupiter = acceleration(r_jupiter, r_sun, M_sun) + acceleration(r_jupiter, r_earth, M_earth)

    # Update velocities
    v_sun += 0.5 * (a_sun + new_a_sun) * dt
    v_earth += 0.5 * (a_earth + new_a_earth) * dt
    v_jupiter += 0.5 * (a_jupiter + new_a_jupiter) * dt

    # Update accelerations
    a_sun = new_a_sun
    a_earth = new_a_earth
    a_jupiter = new_a_jupiter

    # Store positions
    sun_positions.append(r_sun.copy())
    earth_positions.append(r_earth.copy())
    jupiter_positions.append(r_jupiter.copy())

    # Energy
    def energy(r1, r2, m1, m2):
        return -G * m1 * m2 / np.linalg.norm(r1 - r2)

    KE = (0.5 * M_sun * np.linalg.norm(v_sun)**2 +
          0.5 * M_earth * np.linalg.norm(v_earth)**2 +
          0.5 * M_jupiter * np.linalg.norm(v_jupiter)**2)

    PE = (energy(r_sun, r_earth, M_sun, M_earth) +
          energy(r_sun, r_jupiter, M_sun, M_jupiter) +
          energy(r_earth, r_jupiter, M_earth, M_jupiter))

    energies.append(KE + PE)

# Convert to arrays
sun_positions = np.array(sun_positions)
earth_positions = np.array(earth_positions)
jupiter_positions = np.array(jupiter_positions)
energies = np.array(energies)


# Barycenter Frame
barycenter = (
    M_sun * sun_positions +
    M_earth * earth_positions +
    M_jupiter * jupiter_positions
) / (M_sun + M_earth + M_jupiter)

sun_positions = (sun_positions - barycenter) / AU
earth_positions = (earth_positions - barycenter) / AU
jupiter_positions = (jupiter_positions - barycenter) / AU


# Plot Setup 
fig, (ax, ax_energy) = plt.subplots(1, 2, figsize=(14, 6))

fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax_energy.set_facecolor('black')

# Star field
np.random.seed(1)
ax.scatter(np.random.uniform(-6,6,300),
           np.random.uniform(-6,6,300),
           color='white', s=1, alpha=0.6)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

# Planets
sun_dot, = ax.plot([], [], 'o', color='yellow', markersize=10)
earth_dot, = ax.plot([], [], 'o', color='blue', markersize=5)
jupiter_dot, = ax.plot([], [], 'o', color='orange', markersize=7)

earth_trail, = ax.plot([], [], color='cyan', alpha=0.6)
jupiter_trail, = ax.plot([], [], color='orange', alpha=0.4)

# Text
text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='white', family='monospace')

ax.set_title("Sun–Earth–Jupiter Simulation", color='white')
ax.set_xlabel("X (AU)", color='white')
ax.set_ylabel("Y (AU)", color='white')
ax.tick_params(colors='white')
ax.set_aspect('equal')

# Energy plot
energy_line, = ax_energy.plot([], [], color='white')
ax_energy.set_title("Total Energy", color='white')
ax_energy.tick_params(colors='white')


# Animation
def update(frame):

    sun_dot.set_data([sun_positions[frame,0]], [sun_positions[frame,1]])
    earth_dot.set_data([earth_positions[frame,0]], [earth_positions[frame,1]])
    jupiter_dot.set_data([jupiter_positions[frame,0]], [jupiter_positions[frame,1]])

    start = max(0, frame-300)

    earth_trail.set_data(earth_positions[start:frame,0], earth_positions[start:frame,1])
    jupiter_trail.set_data(jupiter_positions[start:frame,0], jupiter_positions[start:frame,1])

    text.set_text(
        f"Earth: ({earth_positions[frame,0]:.2f}, {earth_positions[frame,1]:.2f}) AU\n"
        f"Jupiter: ({jupiter_positions[frame,0]:.2f}, {jupiter_positions[frame,1]:.2f}) AU"
    )

    energy_line.set_data(range(frame), energies[:frame])
    ax_energy.relim()
    ax_energy.autoscale_view()

    return sun_dot, earth_dot, jupiter_dot, earth_trail, jupiter_trail, text, energy_line

ani = FuncAnimation(fig, update, frames=len(sun_positions), interval=15, blit=False)

plt.tight_layout()
plt.show()