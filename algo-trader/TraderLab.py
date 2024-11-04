import json

class TraderLab:

    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as file:
            config = json.load(file)

        # Set attributes
        for key, value in config.items():
            setattr(self, key, value)
