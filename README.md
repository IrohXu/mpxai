# mpxai

### Tony Li-Geng, Xu Cao, Wenqian Ye

MPX AI models.   

### Git clone or update the code   

If you did not download (git clone) it:   

```
git clone git@github.com:IrohXu/mpxai.git
```

else, go the folder and:   

```
git pull
```

### Install in HPC    

#### launch the GPU   
```
srun --partition=gpu4_dev --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=16G --pty bash    
```

#### Load three module    
```
module load miniconda3/gpu/4.9.2
module load git/2.17.0 
module load condaenvs/gpu/pytorch_lightning
```

#### Install python package    
```
pip install -r requirements.txt
```

### The first stage training (public dataset)     

```
python pretrain.py --dataset_path /gpfs/home/xc2057/monkeypox/nyu_monkeypox_dataset
```


### The second stage training (public dataset)     
```
python finetune.py --pretrained_model_path ./ce_pretrained_model.pth --dataset_path /gpfs/home/xc2057/monkeypox/nyu_monkeypox_dataset
```

