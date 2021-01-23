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
