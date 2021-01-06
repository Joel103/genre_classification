# load model config
def load_config(config_path="config.json", verbose=1):
    import json
    with open(config_path, "r") as config_file:
        config_data = json.load(config_file)

    # extract parameter classes
    data_parameter = config_data["data_parameter"]
    model_parameter = config_data["model_parameter"]
    training_parameter = config_data["training_parameter"]

    # show content of config
    if verbose:
        print(json.dumps(config_data, indent=2, sort_keys=True))
        
    return data_parameter, model_parameter, training_parameter

def save_config(data_parameter, model_parameter, training_parameter, network):
    # writing config file into model folder
    import json
    new_config = {"data_parameter": data_parameter, 
                  "model_parameter": model_parameter, 
                  "training_parameter": training_parameter}
    with open(network.config_path, 'w+') as config:
        config.write(json.dumps(new_config, sort_keys=True, indent=2))
    print(f"Model folder created and config saved: {network.config_path}")
    
def allow_growth():
    import tensorflow as tf
    # Copied from: https://tensorflow.google.cn/guide/gpu?hl=en#limiting_gpu_memory_growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                  tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
