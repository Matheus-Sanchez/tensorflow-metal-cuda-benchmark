# Comparativo TensorFlow: Mac Metal vs. Windows/WSL2 CUDA

- Mac: TensorFlow `2.18.1`, tensorflow-metal `1.2.0`
- Windows/WSL2: TensorFlow `2.21.0`, CUDA: `NVIDIA RTX A2000 12GB, 596.51, 12282 MiB`

| Teste | Mac (mediana) | Windows/WSL2 (mediana) | Razão Mac/Windows |
|---|---:|---:|---:|
| matmul_fp32 (4096×4096) | 47.4441 ms | 10.1791 ms | 4.66× |
| tiny_cnn_train_resident_batch (batch 32) | 3.7815 ms | 1.8653 ms | 2.03× |
| tiny_cnn_train_resident_batch (batch 64) | 3.7060 ms | 1.9213 ms | 1.93× |
| tiny_cnn_train_resident_batch (batch 128) | 3.9019 ms | 2.0259 ms | 1.93× |
| tiny_cnn_train_resident_batch (batch 256) | 4.4148 ms | 2.2422 ms | 1.97× |

Uma razão Mac/Windows maior que 1 indica que a execução no Mac demorou mais neste workload específico. A razão não mede desempenho geral de hardware.
