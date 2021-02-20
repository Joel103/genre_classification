## Open Add-Ons

1. Integrate classifier model into training loop. ✅ 
2. Build Inference pipeline: embed -> song-recognition. ✅ 
3. Inspect & maybe implement Transfer Learning (for spectrograms?).
4. Prepare slides. ✅ 
5. Build ResNet-Model. ✅ 
6. Test different dataset. ✅
7. Extending scope lenght of each image (1 image should have longer temporal view than ~3 sec.).

----

# Code

## How to checkout
* ```git submodule update --init```
* ```git submodule update -r```

## How to prepare
* ```pip install -r requirements.txt```
* ```export WANDB_PROJECT=<desired project>```
* For local testing only please also use:
 ```export WANDB_MODE=dryrun```
* For synced training (online):
 ```wandb login```

## How to execute
* Execute "Data_Preprocessing.ipynb" once (1 dataset)
* Execute "Training.ipynb" once per training
* Execute ? once per model to evalute embedding and classify genres/songs

## W&B alternative login for CI/CD:
* ```export WANDB_ENTITY=dl4aed```
* ```export WANDB_API_KEY=<your api key>```

## W&B extras:
* ```export WANDB_NOTEBOOK_NAME=Training```
