# Relatório consolidado: TensorFlow Metal vs. CUDA

**Atualizado em:** 17/09/2026
**Branch de consolidação:** `results/consolidated`
**Fontes:** `results/mac` e `results/windows`

## Escopo e validade da comparação

Os dois resultados foram validados pelo comando `compare`: usam FP32, XLA desativado, matriz 4096×4096, 10 aquecimentos, 100 operações/passos por repetição, cinco repetições e a mesma CNN (`Conv2D(16)-MaxPool-Conv2D(32)-GAP-Dense(10)`) com Adam. As imagens sintéticas e os rótulos foram criados uma única vez na GPU, antes da medição.

| Plataforma | Branch de origem | Execução | Situação |
|---|---|---|---|
| iMac 24” 2024, Apple M4 / Metal | `results/mac` | `results/tf-benchmark-20260916T221446-0300-mac-m4/` | Consolidada |
| Windows no WSL2, RTX A2000 / CUDA | `results/windows` | `results/tf-benchmark-20260917T124917-0300-wsl-rtx-a2000/` | Consolidada |

O teste mede a execução GPU do TensorFlow, e não inclui download de dados, `tf.data`, cópias de host para dispositivo, validação, checkpoints ou callbacks. Portanto, ele não representa sozinho a duração de um experimento completo.

## Ambientes registrados

| Item | Mac | Windows/WSL2 |
|---|---|---|
| Hardware | iMac 24” (2024), Apple M4, GPU integrada de 10 núcleos, 16 GB de memória unificada | NVIDIA RTX A2000 12 GB dedicada |
| Sistema | macOS 15.5, Darwin 24.5.0, ARM64 | Linux 6.18.33.2-microsoft-standard-WSL2, x86_64 |
| Backend GPU confirmado | `tensorflow-metal` 1.2.0, `/device:GPU:0` | CUDA 12.5.1 / cuDNN 9, `/device:GPU:0` |
| TensorFlow | 2.18.1 | 2.21.0 |
| XLA | Desativado | Desativado |

Há uma diferença de versão do TensorFlow entre as execuções. Assim, este é um comparativo dos ambientes realmente usados (hardware + backend + runtime), não uma medição que isole apenas a arquitetura do hardware.

## Resultado individual — Mac M4 / TensorFlow Metal

| Teste | Tempo mediano | Throughput mediano |
|---|---:|---:|
| Matmul FP32 4096×4096 | 47,4441 ms/op | 2.896,9 GFLOP/s |
| CNN, batch 32 | 3,7815 ms/passo | 264,4 passos/s; 8.462,1 imagens/s |
| CNN, batch 64 | 3,7060 ms/passo | 269,8 passos/s; 17.269,4 imagens/s |
| CNN, batch 128 | 3,9019 ms/passo | 256,3 passos/s; 32.804,2 imagens/s |
| CNN, batch 256 | 4,4148 ms/passo | 226,5 passos/s; 57.986,5 imagens/s |

Arquivos de evidência: `metadata.json`, `summary.json` e `samples.csv` em `results/tf-benchmark-20260916T221446-0300-mac-m4/`.

## Resultado individual — RTX A2000 / TensorFlow CUDA

| Teste | Tempo mediano | Throughput mediano |
|---|---:|---:|
| Matmul FP32 4096×4096 | 10,1791 ms/op | 13.502,0 GFLOP/s |
| CNN, batch 32 | 1,8653 ms/passo | 536,1 passos/s; 17.155,8 imagens/s |
| CNN, batch 64 | 1,9213 ms/passo | 520,5 passos/s; 33.310,1 imagens/s |
| CNN, batch 128 | 2,0259 ms/passo | 493,6 passos/s; 63.180,4 imagens/s |
| CNN, batch 256 | 2,2422 ms/passo | 446,0 passos/s; 114.172,1 imagens/s |

Arquivos de evidência: `metadata.json`, `summary.json` e `samples.csv` em `results/tf-benchmark-20260917T124917-0300-wsl-rtx-a2000/`.

## Comparação automática

| Teste | Mac M4 / Metal | RTX A2000 / CUDA | Razão Mac/Windows |
|---|---:|---:|---:|
| Matmul FP32 4096×4096 | 47,4441 ms/op | 10,1791 ms/op | 4,66× |
| CNN, batch 32 | 3,7815 ms/passo | 1,8653 ms/passo | 2,03× |
| CNN, batch 64 | 3,7060 ms/passo | 1,9213 ms/passo | 1,93× |
| CNN, batch 128 | 3,9019 ms/passo | 2,0259 ms/passo | 1,93× |
| CNN, batch 256 | 4,4148 ms/passo | 2,2422 ms/passo | 1,97× |

Razão maior que 1 significa que o Mac levou mais tempo naquele workload. O artefato gerado pelo comando de comparação está em `results/tf-comparison-20260917T130256-0300/`.

## Conclusão técnica

- No `matmul` grande, o ambiente CUDA foi 4,66× mais rápido. Isso se aproxima da diferença de 5–6× observada em alguns treinos, mas não a reproduz integralmente.
- Na CNN pequena com lote já residente na GPU, o ambiente CUDA foi entre 1,93× e 2,03× mais rápido. Logo, para esse núcleo de treino isolado, a diferença não é de 5–6×.
- Se o treino real permanece 5–6× mais lento no Mac, a parcela adicional provavelmente está fora deste microbenchmark: pipeline `tf.data`, augmentations, I/O, tamanho/forma do modelo, validação, checkpoints, callbacks, ou operações que caem na CPU. A revisão do código do treino é necessária para localizar essa parcela.
- Os 16 GB unificados do M4 e os 12 GB de VRAM dedicada da RTX A2000 não são métricas equivalentes. Além da capacidade, diferem o tipo de memória, o compartilhamento com a CPU e a pilha Metal versus CUDA/cuDNN.

Para reprodução, use `outputs/tensorflow_benchmark.py compare` com os dois diretórios acima; o comando bloqueia comparações com parâmetros incompatíveis e salva CSV, JSON e Markdown.
