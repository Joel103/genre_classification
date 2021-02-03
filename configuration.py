embedding_size = 1024
feature_size = 64
channels = 1
input_shape = (None, feature_size, channels)
gaussian_noise = 0.1

config = {
    "model_parameter": {
        "encoder_input": {
            "class_name": "EncoderInput",
            "config": {
                "layers": {
                    0: {"class_name": "InputLayer",
                        "config":{ "input_shape": input_shape, },
                    },
                    1: {"class_name": "GaussianNoise",
                        "config":{ "stddev": gaussian_noise}
                    },
                    2: {"class_name": "Reshape",
                        "config":{ "target_shape": (-1, feature_size, feature_size, channels), },
                    },
                },
                "name": "Input",
            },
        },
        "encoder": {
            "class_name": "Encoder",
            "config" : {
                "layers": {
                    0: {"class_name": "InputLayer",
                        "config":{ "input_shape": (feature_size, feature_size, channels), },
                    },
                    0.1: {"class_name": "BatchNormalization",
                          "config": {},
                    },
                    1: {"class_name": "Conv2DBlock",
                        "config":{
                             "num_filters": 32,
                             "kernel_size": 3,
                             "conv_stride": 1,
                             "pool_size": 2,
                             "pool_stride": 2,
                             "spatial_dropout": True,
                        },
                    },
                    2: {"class_name": "Conv2DBlock",
                        "config":{
                             "num_filters": 64,
                             "kernel_size": 3,
                             "conv_stride": 1,
                             "pool_size": 2,
                             "pool_stride": 2,
                             "spatial_dropout": True,
                        },
                    },
                    3: {"class_name": "Conv2DBlock",
                        "config":{
                             "num_filters": 128,
                             "kernel_size": 3,
                             "conv_stride": 1,
                             "pool_size": 2,
                             "pool_stride": 2,
                        },
                    },
                    4: {"class_name": "Conv2DBlock",
                        "config":{
                             "num_filters": embedding_size,
                             "kernel_size": 3,
                             "conv_stride": 1,
                             "pool_size": 2,
                             "pool_stride": 2,
                        },
                    },
                    5: {"class_name": "GlobalAveragePooling2D",
                        "config":{
                        },
                    },
                },
            },
        },
        "decoder": {
            "class_name": "Decoder",
            "config": {
                "layers": {
                    0: {"class_name": "InputLayer",
                        "config":{ "input_shape": (embedding_size), },
                    },
                    0.1: {"class_name": "Reshape",
                        "config":{ "target_shape": (1, 1, embedding_size), },
                    },
                    1: {"class_name": "Conv2DTransposeBlock",
                        "config":{
                             "num_filters": 128,
                             "kernel_size": 3,
                             "conv_stride": 1,
                             "pool_size": 2,
                        },
                    },
                    2: {"class_name": "Conv2DTransposeBlock",
                        "config":{
                             "num_filters": 64,
                             "kernel_size": 3,
                             "conv_stride": 1,
                             "pool_size": 2,
                        },
                    },
                    3: {"class_name": "Conv2DTransposeBlock",
                        "config":{
                             "num_filters": 32,
                             "kernel_size": 3,
                             "conv_stride": 1,
                             "pool_size": 2,
                        },
                    },
                    4: {"class_name": "Conv2DTransposeBlock",
                        "config":{
                             "num_filters": 32,
                             "kernel_size": 3,
                             "conv_stride": 1,
                             "pool_size": 2,
                        },
                    },
                },
            },
        },
        "finalizer": {
            "layers" : {

            },
        },
        "output": {
            "class_name": "Output",
            "config": {
                "layers": {
                    0: {"class_name": "InputLayer",
                        "config":{ "input_shape": (None, feature_size, feature_size, channels), },
                    },
                    1: {"class_name": "Reshape",
                        "config":{ "target_shape": (-1, feature_size, channels), },
                    },
                },
                "name": "Output",
            },
        },
    },
}