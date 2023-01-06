import yaml

def pos_to_pix(x, y, screen_width, screen_height):
    x += screen_width/2
    y = screen_height - y - 1
    y -= screen_height/2
    return x, y

def load_config(fp) -> dict:
    with open(fp) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255, 0 ,0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
size = (300, 300)