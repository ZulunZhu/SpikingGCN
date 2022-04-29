> Install:

require: python 3.6+, pytorch and some common packages.
```
conda create -n py36 python=3.6
conda activate py36
pip install graphgallery==0.7.2 spikingjelly pandas

pip install thop scikit-learn
cd path_to_spikingGCN/handcode/
python run_snn.py
```

+ In case are prompted that other dependent packages are missing, can install it with: pip install xxx.
+ Set parameters in models_conf.json, such as device": "cuda:0"