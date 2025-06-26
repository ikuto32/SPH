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
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        world.step(dt)
        positions = world.get_positions()

        screen.fill((0, 0, 0))
        for x, y in positions:
            pygame.draw.circle(
                screen,
                (255, 255, 255),
                (int(x * scale), int(height - y * scale)),
                2,
            )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
