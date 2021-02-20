import argparse
import json
from utils import update

# parsing default parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="dl4aed argument parser")
    #parser.add_argument("-e", "--epochs", type=int, help="number of epochs to train")
    #parser.add_argument("-l", "--learning_rate", type=float, help="constant learning rate")
    #parser.add_argument("-b", "--batch_size", type=int, help="training batch size")
    parser.add_argument("-c", "--custom", type=str, help="string which is converted to dict to override any setting")

    arguments = parser.parse_known_args()[0].__dict__
    if not arguments["custom"] is None:
        custom = json.loads(arguments["custom"])
        arguments = update(arguments, custom)
        del(arguments["custom"])
    
    return arguments

if __name__ == "__main__":
    # example usage
    from utils import load_config
    _, model_parameter, training_parameter = load_config(verbose=0)
    parsed_arguments, _ = parse_arguments()