## Open Add-Ons

1. Integrate classifier model into training loop.
2. Build Inference pipeline: embed -> song-recognition.
3. Inspect & maybe implement Transfer Learning (for spectrograms?).
4. Prepare slides.
5. Build ResNet-Model.

----

# Code

## How to checkout
* ```git submodule update --init```
* ```git submodule update -r```

## How to prepare
* ```pip install -r requirements.txt```
* ```wandb login```
* ```export WANDB_PROJECT=training```
* For local testing please also use:
 ```export WANDB_MODE=dryrun```
 
## How to execute
* 

## W&B alternative login for CI/CD:
* ```export WANDB_ENTITY=dl4aed```
* ```export WANDB_API_KEY=<your api key>```

## W&B extras:
* ```export WANDB_NOTEBOOK_NAME=main```
