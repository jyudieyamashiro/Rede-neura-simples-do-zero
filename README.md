# Rede Neural do Zero com PyTorch 🧠🔥

Este projeto implementa uma rede neural simples **do zero** usando `PyTorch`, treinada para reconhecer dígitos manuscritos do famoso dataset **MNIST**.

## 🧰 Tecnologias utilizadas

- Python
- PyTorch
- torchvision
- matplotlib
- numpy

## 📦 Estrutura do Projeto

- `MNIST_data/`: pasta onde o dataset é baixado automaticamente
- `Rede neural do zero.py`: código principal com definição do modelo, treino e validação

## 🧠 Arquitetura da Rede

- Entrada: 784 neurônios (28x28 pixels)
- Camada Oculta 1: 128 neurônios + ReLU
- Camada Oculta 2: 64 neurônios + ReLU
- Saída: 10 neurônios + LogSoftmax

## 📈 Treinamento

- Otimizador: SGD com `lr=0.01` e `momentum=0.05`
- Critério: `Negative Log Likelihood Loss (NLLLoss)`
- Épocas: 10
- Tempo de treino: ~1,3 minutos

### 🟢 Exemplo de saída:

Epoch 1 - Perda resultante: 1.7524175115231513
Epoch 2 - Perda resultante: 0.5661082095714774
Epoch 3 - Perda resultante: 0.39996617254036576
Epoch 4 - Perda resultante: 0.3475752471686045
Epoch 5 - Perda resultante: 0.31678199331198675
Epoch 6 - Perda resultante: 0.29317812751065186
Epoch 7 - Perda resultante: 0.2731502692121814
Epoch 8 - Perda resultante: 0.2557229829797231
Epoch 9 - Perda resultante: 0.23958897902401907
Epoch 10 - Perda resultante: 0.2249544599075625

Tempo de treino (em minutos) = 1.3286304593086242
Total de imagens testadas = 10000


## 🖼️ Visualização de amostra

O código exibe uma imagem de exemplo do dataset MNIST no início do script para familiarização com os dados.

✨ Resultado
O modelo atinge aproximadamente 94% de acurácia após apenas 10 épocas, demonstrando a eficácia de uma rede simples bem configurada.

📚 Créditos
Este projeto foi desenvolvido como parte de um bootcamp de Machine Learning.


