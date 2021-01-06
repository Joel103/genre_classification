import argparse

# parsing default parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description="dl4aed argument parser")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs to train")
    parser.add_argument("-l", "--learning_rate", type=float , help="constant learning rate")
    parser.add_argument("-b", "--batch_size", type=int , help="training batch size")
    return parser.parse_known_args()

# overwriting existing config options
def overwrite_config(parsed_arguments, data_parameter, model_parameter, training_parameter):
    arguments = vars(parsed_arguments)
    for argument in arguments:
        if not arguments[argument] is None:
            if argument in data_parameter:
                data_parameter[argument] = arguments[argument]
            elif argument in model_parameter:
                model_parameter[argument] = arguments[argument]
            elif argument in training_parameter:
                training_parameter[argument] = arguments[argument]

if __name__ == "__main__":
    # example usage
    from utils import load_config
    _, model_parameter, training_parameter = load_config(verbose=0)
    parsed_arguments, _ = parse_arguments()
    overwrite_config(parsed_arguments, model_parameter, training_parameter)
