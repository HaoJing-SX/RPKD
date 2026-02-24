# Reflectance Prediction-Based Knowledge Distillation for Robust 3D Object Detection in Compressed Point Clouds

This is the official implementation of "Reflectance Prediction-Based Knowledge Distillation for Robust 3D Object Detection in Compressed Point Clouds". This repository is based on [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) and [`SparseKD`](https://github.com/CVMI-Lab/SparseKD).

**Abstract**: Regarding intelligent transportation systems, low-bitrate transmission via lossy point cloud compression is vital for facilitating real-time collaborative perception among connected agents, such as vehicles and infrastructures, under restricted bandwidth. In existing compression transmission systems, the sender lossily compresses point coordinates and reflectance to generate a transmission code stream, which faces transmission burdens from reflectance encoding and limited detection robustness due to information loss. To address these issues, this paper proposes a 3D object detection framework with reflectance prediction-based knowledge distillation (RPKD). We compress point coordinates while discarding reflectance during low-bitrate transmission, and feed the decoded non-reflectance compressed point clouds into a student detector. The discarded reflectance is then reconstructed by a geometry-based reflectance prediction (RP) module within the student detector for precise detection. A teacher detector with the same structure as the student detector is designed for performing reflectance knowledge distillation (RKD) and detection knowledge distillation (DKD) from raw to compressed point clouds. Our cross-source distillation training strategy (CDTS) equips the student detector with robustness to low-quality compressed data while preserving the accuracy benefits of raw data through transferred distillation knowledge. Experimental results on the KITTI and DAIR-V2X-V datasets demonstrate that our method can boost detection accuracy for compressed point clouds across multiple code rates.

<img src="docs/Overall3.png" align="center" width="100%">

## License

`SMS` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet)
[`SparseKD`](https://github.com/CVMI-Lab/SparseKD)

## Citation 

If you find this project useful in your research, please consider cite:

```
@ARTICLE{11322695,
  author={Jing, Hao and Wang, Anhong and Zhang, Yifan and Bu, Donghan and Hou, Junhui},
  journal={IEEE Transactions on Image Processing}, 
  title={Reflectance Prediction-Based Knowledge Distillation for Robust 3D Object Detection in Compressed Point Clouds}, 
  year={2026},
  volume={35},
  number={},
  pages={85-97},
  keywords={Image coding;Reflectivity;Point cloud compression;Three-dimensional displays;Object detection;Detectors;Training;Accuracy;Robustness;Feature extraction;Compressed point clouds;3D object detection;knowledge distillation;reflectance prediction},
  doi={10.1109/TIP.2025.3648203}}
```

## Email 

If you have any questions, please contact jinghao@tyust.edu.cn.
