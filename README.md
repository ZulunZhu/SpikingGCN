This repository includes the source code and appendix of "**Spiking Graph Convolutional Networks**" which will be published in IJCAI 2022.

## 🗻 Install:

require: python 3.6+, pytorch and some common packages.

```
conda create -n py36 python=3.6
conda activate py36
pip install graphgallery==0.7.2 spikingjelly pandas

pip install thop scikit-learn
```

<br/>

- In case are prompted that other dependent packages are missing, can install it with: pip install xxx.
- Set parameters in models_conf.json, such as device": "cuda:0"
  
  <br/>

## 🏝️ **Run**

```
cd path_to_spikingGCN/handcode/
python run_snn.py
```


  Also you can run the SpikingGCN.ipynb notebook.


<br/>

- For other baseline models, you can

```
cd gnn_models/
python run_sgc.py
```

<br/>

- For the active learning test, you can

```
cd active_snn/
```


  and test the al_snn.ipynb.


<br/>

- For the image classification test, you can

```
cd mnist_snn/
```


  and test superpixel_MNIST.ipynb or MNIST.ipynb.


<br/>

- Here exist some other experiments we ever tried, like the robustness and bayesian neural networks, which can be explored in the future. You can view them in attack_snn/ and bayesianSNN/ .

<br/>

## 😘 Acknowledgement

This project is motivated by [GraphGallery](https://github.com/EdisonLeeeee/GraphGallery.git), [spikingjelly](https://github.com/fangwei123456/spikingjelly.git) and [LISNN](https://github.com/Delver-of-Squeakrets/LISNN.git), etc., and the original implementations of the authors, thanks for their excellent works!
