import pygame
import time
import random

# Define the dimensions of the window
window_width, window_height = 800, 600

# Number of charging stations and ports
C = 5
ports_per_station = 2

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Charging Station Environment")

# Load images for charging station and electric vehicle
charging_station_img = pygame.image.load("cs.jpg")
electric_vehicle_img = pygame.image.load("ev.png")

# Function to add a new electric vehicle near a charging station
def add_vehicle():
    station = random.choice(charging_stations)
    x, y = station
    offset_x, offset_y = random.randint(-50, 50), random.randint(-50, 50)
    vehicles.append((x + offset_x, y + offset_y))

# Divide the screen into C parts
charging_stations = [(window_width * (i + 1) // (C + 1), window_height // 2) for i in range(C)]
vehicles = []

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Add new vehicle with a certain probability (adjust the value as needed)
    if random.random() < 0.03:
        add_vehicle()

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw charging stations
    for station in charging_stations:
        x, y = station
        screen.blit(charging_station_img, (x - charging_station_img.get_width() // 2, y - charging_station_img.get_height() // 2))

    # Draw vehicles
    for vehicle in vehicles:
        x, y = vehicle
        screen.blit(electric_vehicle_img, (x - electric_vehicle_img.get_width() // 2, y - electric_vehicle_img.get_height() // 2))

    pygame.display.flip()
    time.sleep(0.1)  # Adjust the sleep time to control the speed of the simulation

pygame.quit()
