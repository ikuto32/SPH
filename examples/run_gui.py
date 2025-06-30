import time
import pygame
from _sph import PyWorld
import _sph


def main():
    world = PyWorld()
    gpu_search = getattr(_sph, "hash2d_enabled", False)
    scale = 40
    width = int(world.width() * scale)
    height = int(world.height() * scale)

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    dt = 1 / 60.0
    running = True
    paused = False
    step_once = False
    force_radius = 2.0
    force_strength = 10.0

    while running:
        start_time = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_s:
                    step_once = True
                elif event.key == pygame.K_r:
                    world = PyWorld()
                    paused = False
                    step_once = False
        
        left, middle, right = pygame.mouse.get_pressed(num_buttons=3)
        if left or right:
            x, y = pygame.mouse.get_pos()
            world_x = x / scale
            world_y = y / scale
            strength = force_strength if left else -force_strength
            world.set_interaction_force(world_x, world_y, force_radius, strength)
        else:
            world.delete_interaction_force()

        if not paused or step_once:
            world.step(dt)
            step_once = False
        positions = world.get_positions()
        proc_time = (time.perf_counter() - start_time) * 1000.0

        screen.fill((0, 0, 0))
        for x, y in positions:
            pygame.draw.circle(
                screen,
                (255, 255, 255),
                (int(x * scale), int(y * scale)),
                2,
            )
        if paused:
            text = f"Paused | Time: {proc_time:.2f} ms"
        else:
            text = f"FPS: {clock.get_fps():.2f} | Time: {proc_time:.2f} ms"
        mode = "GPU" if gpu_search else "CPU"
        text += f" | Neighbour search: {mode}"
        img = font.render(text, True, (255, 0, 0))
        screen.blit(img, (10, 10))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
