# ResGEM
Official code for the TVCG paper "ResGEM: Multi-scale Graph Embedding Network for Residual Mesh Denoising"

### Environment
Pytorch and torch_geometric are required.

### Usage
We temporally provide PREPROCESSED Synthetic and checkpoint data of the first normal regression iteration, which can be run by the following command:
```
python infer.py --testset ./data/test --data_type Synthetic --graph_type DENSE_FACE --resume ./ckpt/Synthetic_nn.pth --save_dir ./data/results
```

### Citation
```
@ARTICLE{ResGEM2024,
  author={Zhou, Ziqi and Yuan, Mengke and Zhao, Mingyang and Guo, Jianwei and Yan, Dong-Ming},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={ResGEM: Multi-scale Graph Embedding Network for Residual Mesh Denoising}, 
  year={2024},
  pages={1-17},
  doi={10.1109/TVCG.2024.3378309}}
```
