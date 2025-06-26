import numpy as np
import matplotlib.pyplot as plt
from _sph import PyWorld


def main():
    world = PyWorld()
    dt = 1.0 / 60.0
    # advance the world for some steps
    for _ in range(120):
        world.step(dt)
    pos = world.get_positions()
    vel = world.get_velocities()
    speed = np.linalg.norm(vel, axis=1)

    plt.figure(figsize=(8, 4))
    scatter = plt.scatter(pos[:, 0], pos[:, 1], c=speed, cmap="viridis", s=10)
    plt.colorbar(scatter, label="speed")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle speeds after 2 seconds")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig("snapshot.png", dpi=150)


if __name__ == "__main__":
    main()
