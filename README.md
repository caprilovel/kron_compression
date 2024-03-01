# Kronecker product decomposition  

Kronecker product decomposition in different dataset and model.

## Get Start
You can training the Kronecker Model by using `main.py`

```bash
python main.py
```

if you want to use MNIST Dataset Using LeNet
```bash
python main.py -dataset mnist -model LeNet -kron True
```
if you want to use VIT base 16  Using CIFAR-10 
```bash
python main.py -dataset cifar10 -model vit_base_16 -kron True
```

