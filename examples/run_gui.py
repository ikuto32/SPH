import time
import pygame
from _sph import PyWorld


def main():
    world = PyWorld()
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

    # store current query position for neighbour highlight
    query_pos = None
    neighbour_indices = []
    hash_indices = []

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
            query_pos = (world_x, world_y)
        else:
            world.delete_interaction_force()
            query_pos = None
            neighbour_indices = []
            hash_indices = []

        if not paused or step_once:
            world.step(dt)
            step_once = False
        positions = world.get_positions()
        if query_pos is not None:
            # query neighbours using the C++ implementation
            neighbour_indices = list(world.query_neighbors(query_pos[0], query_pos[1]))
            hash_indices = list(world.query_spatial_hash(query_pos[0], query_pos[1]))
        proc_time = (time.perf_counter() - start_time) * 1000.0

        screen.fill((0, 0, 0))
        for idx, (x, y) in enumerate(positions):
            if idx in neighbour_indices:
                color = (0, 255, 0)
            elif idx in hash_indices:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            pygame.draw.circle(
                screen,
                color,
                (int(x * scale), int(y * scale)),
                2,
            )
        if query_pos is not None:
            pygame.draw.circle(
                screen,
                (0, 255, 0),
                (int(query_pos[0] * scale), int(query_pos[1] * scale)),
                int(world.smoothing_radius() * scale),
                1,
            )
        if paused:
            text = f"Paused | Time: {proc_time:.2f} ms"
        else:
            text = f"FPS: {clock.get_fps():.2f} | Time: {proc_time:.2f} ms"
        img = font.render(text, True, (255, 0, 0))
        screen.blit(img, (10, 10))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
