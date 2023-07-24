from dip.boardgames.saboteur.Game import Game
from dip.boardgames.saboteur.GUI import *
from dip.boardgames.saboteur.Camera import *

if __name__ == "__main__":
    # UNCOMMENT to setup and configure camera before starting game
    # camera = Camera(device=0)
    # camera.setup()


    # Start the game
    saboteur = Game()
    saboteur.start(cam_device_id=0)
