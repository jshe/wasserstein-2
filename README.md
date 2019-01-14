# W2GAN: Recovering an Optimal Map with a GAN

Code accompanying paper of the same name.

## Dependencies

```
torch (0.4.1)
numpy (1.14.3)
h5py (2.7.1)
torchvision (0.2.1)
scikit-learn (sklearn) (0.19.1)
matplotlib (2.2.2)
python (3.6)
tensorboardX (optional, remove dependency if not used)
```

## Experiments

### 2D (exp_2d)

```
# 4 gaussians
python main.py --solver=w2 --gen=1 --data=4gaussians
# swissroll
python main.py --solver=w2 --gen=1 --data=swissroll
# checkerboard
python main.py --solver=w2 --gen=1 --data=checkerboard
```

### Multivariate Gaussian ⟶ MNIST (exp_mvg)

```
python main.py --solver=w2
```

### Domain Adaptation: MNIST ⟷ USPS (exp_da)

```
# usps -> mnist
python main.py --solver=w2 --direction=usps-mnist
# mnist -> usps
python main.py --solver=w2 --direction=mnist-usps
```
## Acknowledgments

* https://github.com/mikigom/large-scale-OT-mapping-TF.git
* https://github.com/igul222/improved_wgan_training.git
