# Configuração técnica do Mac para o experimento TensorFlow

**Coleta original:** 16/09/2026, 17:40 (BRT)  
**Escopo:** características relevantes para o comparativo TensorFlow Metal versus TensorFlow CUDA.  
**Privacidade:** números de série, UUIDs, nome do computador, identificadores de rede e de acessórios foram removidos.

## Resumo

O equipamento é um **iMac de 24 polegadas (2024)**, com **Apple M4**, CPU de 10 núcleos (4 de desempenho e 6 de eficiência), GPU integrada de 10 núcleos, **16 GB de memória unificada** e SSD interno de **512 GB**. Na coleta, executava **macOS 15.5 (24F74)**, arquitetura ARM64.

| Componente | Configuração observada |
|---|---|
| Modelo | iMac 24", 2024, quatro portas Thunderbolt 4/USB 4 |
| Identificador da família | Mac16,3 |
| SoC | Apple M4 |
| CPU | 10 núcleos: 4 de desempenho + 6 de eficiência |
| GPU | Apple M4 integrada, 10 núcleos |
| Neural Engine | 16 núcleos |
| Memória | 16 GB LPDDR5 Micron, memória unificada |
| Largura de banda da memória | Até 120 GB/s, conforme especificação Apple |
| Armazenamento | SSD Apple interno de 512 GB, TRIM suportado |
| Espaço livre na coleta | Aproximadamente 228 GiB no contêiner APFS |
| Sistema | macOS 15.5, Darwin 24.5.0, ARM64 |
| Energia na coleta | Alimentação AC; Low Power Mode desativado |

## Consequências para o benchmark

- A GPU M4 não possui VRAM dedicada: CPU, GPU e aceleradores usam o mesmo pool de 16 GB. Portanto, a memória reportada pelo macOS/Metal não é diretamente comparável aos 12 GB de GDDR6 da RTX A2000.
- O TensorFlow usa o plug-in `tensorflow-metal` para expor a GPU Apple como dispositivo TensorFlow. O benchmark deve confirmar isso por `tf.config.list_physical_devices('GPU')` e registrar a versão do plug-in, em vez de inferir pelo nome do computador.
- A largura de banda de memória publicada para o M4 é de até 120 GB/s. A RTX A2000 tem memória dedicada e segue o caminho CUDA/cuDNN; o resultado deve ser descrito como comparação dos backends e deste workload, não como uma equivalência direta de núcleos ou percentuais de utilização.
- A coleta não informa versão de Python, TensorFlow, `tensorflow-metal`, nem detalhes do runtime Metal usado no treino. O benchmark captura esses dados automaticamente em `metadata.json`, pois são essenciais para interpretar qualquer diferença.

## Condições de execução recomendadas

1. Manter o iMac em alimentação AC, com Low Power Mode desativado e sem tarefas pesadas concorrentes.
2. Usar a mesma precisão (`float32`), aquecimento, número de iterações e versão do script nos dois computadores.
3. Fazer uma execução regular para o resultado final e, se necessário, outra com `--profile`/`--verify-placement` apenas para diagnóstico.
4. Preservar `metadata.json` junto com os resultados: versões diferentes de TensorFlow e `tensorflow-metal` podem alterar substancialmente o desempenho.

## Limitações da coleta

A coleta original não pôde obter alguns detalhes de `system_profiler`, incluindo enumeração detalhada de Bluetooth, câmera, áudio e painel. Essas limitações não impedem os dois microbenchmarks, mas este documento não deduz temperaturas, frequência de CPU, saúde do SSD ou métricas de GPU que não foram fornecidas.

## Fonte técnica

As especificações do modelo foram conferidas com a [ficha técnica do iMac 24" (2024) da Apple](https://support.apple.com/pt-br/121557). Para o caminho de aceleração TensorFlow/Metal, utilize a documentação da Apple para o [plug-in tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/).
