# Laplacian-Guided Hierarchical Transformer: A Network for Medical Image Segmentation <br> <span style="float: right"><sub><sup></sub></sup></span>

## Train

1) Download the Synapse Dataset from [here](https://drive.google.com/uc?export=download&id=18I9JHH_i0uuEDg-N6d7bfMdf7Ut6bhBi).

2) Run the following code to install the requirements.

    ```python
   pip install -r requirements.txt
    ```

4) Run the below code to train the model on the synapse dataset.

  ```python
    python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5  --batch_size 20 --eval_interval 20 --max_epochs 600 --dst_fast --resume --model_path [MODEL PATH]
  ```
    
   ```
    --root_path    [Train data path]

    --test_path    [Test data path]

    --eval_interval [Evaluation epoch]

    --dst_fast [Optional] [Load all data into RAM for faster training]

    --resume [Optional] [Resume from checkpoint]

    --model_path [Optional] [Provide the path to the latest checkpoint file for loading the model.]
   ```


## Test 

1) Run the below code to test the model on the synapse dataset.
  
    
    ```python
    python test.py --test_path ./data/Synapse/test_vol_h5 --is_savenii --pretrained_path './best_model.pth'
    ```

    ```
    --test_path     [Test data path]
        
    --is_savenii    [Whether to save results during inference]

    --pretrained_path  [Pretrained model path]
    ```
