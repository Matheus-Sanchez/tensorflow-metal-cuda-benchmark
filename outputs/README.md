# Benchmark TensorFlow: Metal no Mac vs. CUDA no Windows/WSL2

O arquivo `tensorflow_benchmark.py` mede apenas dois workloads em FP32:

1. `tf.linalg.matmul` com matrizes `4096×4096`;
2. uma CNN pequena com um lote sintético já criado na GPU.

Ele não baixa datasets e não mede `tf.data`, DataLoader, disco ou cópia CPU→GPU. Isso é intencional: os resultados isolam o backend TensorFlow Metal/CUDA e ajudam a decidir se a diferença observada no treino real vem do pipeline de dados ou do próprio backend.

## Preparação

Use o mesmo ambiente virtual TensorFlow da pesquisa sempre que possível. O script exige Python 3.9+ e TensorFlow; no Mac, a GPU requer o plug-in `tensorflow-metal` da Apple. No Windows, execute dentro do WSL2 com TensorFlow/CUDA e `nvidia-smi` funcionais — TensorFlow recente não oferece CUDA oficialmente no Windows nativo.

No Mac, a instalação normalmente inclui:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow tensorflow-metal
```

No WSL2, use a instalação CUDA documentada pelo TensorFlow, por exemplo:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install 'tensorflow[and-cuda]'
nvidia-smi
```

Antes de medir, confirme que TensorFlow lista uma GPU:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Consulte a documentação oficial do [TensorFlow para instalação via pip](https://www.tensorflow.org/install/pip) e o guia da Apple para o [plug-in tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/) se a GPU não aparecer.

## Executar

Copie o mesmo `tensorflow_benchmark.py` para ambos os computadores. Feche tarefas pesadas e mantenha o iMac ligado à energia.

No Mac:

```bash
python3 tensorflow_benchmark.py run --label mac-m4 --expect metal --output-dir resultados
```

No WSL2, exigindo que a GPU seja a RTX A2000:

```bash
python3 tensorflow_benchmark.py run --label wsl-rtx-a2000 --expect cuda --require-gpu-name "RTX A2000" --output-dir resultados
```

Cada comando cria uma pasta independente contendo:

- `metadata.json`: versões, dispositivos TensorFlow, build, GPU/driver e configurações;
- `samples.csv`: cada repetição individual;
- `summary.json`: médias, medianas e especificação completa;
- `report.md`: tabela pronta para leitura.

Para investigar posicionamento de operações ou gerar um trace TensorBoard, faça uma execução separada (não use essa saída como o resultado final):

```bash
python3 tensorflow_benchmark.py run --label mac-profile --expect metal --verify-placement --profile --output-dir resultados
```

## Comparar os computadores

Depois de copiar os diretórios de resultados para o mesmo computador, execute:

```bash
python3 tensorflow_benchmark.py compare \
  --mac-run resultados/tf-benchmark-AAAAmmddTHHMMSS-0000-mac-m4 \
  --windows-run resultados/tf-benchmark-AAAAmmddTHHMMSS-0000-wsl-rtx-a2000 \
  --output-dir resultados
```

O comando aborta se parâmetros de teste diferirem e cria `comparison.md`, `comparison.csv` e `comparison.json`. Uma razão de tempo Mac/Windows acima de 1 significa somente que o Mac foi mais lento naquele workload TensorFlow específico; não é uma medida geral de desempenho de hardware.

## Auditoria do treino real

O benchmark é independente do treino de pesquisa. Quando o código TensorFlow for disponibilizado, a revisão deve verificar `tf.data` (`cache`, `prefetch`, paralelismo de `map`), device placement, callbacks, validação, checkpoints e se a medição por época inclui etapas fora do treino.
