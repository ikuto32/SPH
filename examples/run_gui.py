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

    dt = 1 / 60.0
    running = True
    force_radius = 2.0
    force_strength = 10.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        left, middle, right = pygame.mouse.get_pressed(num_buttons=3)
        if left or right:
            x, y = pygame.mouse.get_pos()
            world_x = x / scale
            world_y = y / scale
            strength = force_strength if left else -force_strength
            world.set_interaction_force(world_x, world_y, force_radius, strength)
        else:
            world.delete_interaction_force()

        world.step(dt)
        positions = world.get_positions()

        screen.fill((0, 0, 0))
        for x, y in positions:
            pygame.draw.circle(
                screen,
                (255, 255, 255),
                (int(x * scale), int(y * scale)),
                2,
            )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
