# Bibliotecas a serem utilizadas
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.ToTensor() # Definindo a conversão de imagem para tensor

trainset = datasets.MNIST('./MNIST_data/', download=True, transform=transform) # Carrega a parte de treino do dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) # Cria um buffer para pegar os dados por partes

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform) # Carrega a parte da validação do dataset
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True) # Cria um buffer ára pegar os dados por partes

dataiter = iter(trainloader)
imagens, etiquetas = next(dataiter)
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
plt.show()  

print(imagens[0].shape) # Para verificar as dimensões do tensor de cada imagem
print(etiquetas[0].shape) # Para verificar as dimensões do tensor de cada etiqueta

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128) # Camada de entrada, 784 neurônior que se ligam a 128
        self.linear2 = nn.Linear(128, 64) # Camada interna 1, 128 neurônios que se ligam a 64
        self.linear3 = nn.Linear(64, 10) # Camada interna 2, 64 neurônios que se ligam a 10
        
    def forward(self,X):
        X = F.relu(self.linear1(X)) # Função de ativação da camada de entrada para a camada interna 1
        X = F.relu(self.linear2(X)) # Função de ativação da camada de entrada para a camada interna 2
        X = self.linear3(X) # Função de ativação da camada interna 2 para a camada de saida (f(x) = x)
        return F.log_softmax(X, dim=1) # Dados utilizados para calcular a perda
    
def treino(modelo, trainloader, device):

    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.05) # Define a politica de atualização dos pesos e da bias
    inicio = time() # Timer para saber quanto tempo levou o treino

    criterio = nn.NLLLoss() # Definição do critério para calcular a perda
    EPOCHS = 10 # Número de epochs que o algoritmo rodará

    for epoch in range(EPOCHS):
        perda_acumulada = 0 # Inicialização da perda acumulada da epoch em questão

        for imagens, etiquetas in trainloader:

            imagens = imagens.view(imagens.shape[0], -1) # Converter as imagens para vetoers de 28*28
            otimizador.zero_grad() # Zerar os gradientes por conta do ciclo anterior

            output = modelo(imagens.to(device)) # Colocar os dados no modelo
            perda_instantanea = criterio(output, etiquetas.to(device)) # Calcular a perda do epoch

            perda_instantanea.backward() # Back propagation a partir da perda

            otimizador.step() # Atualizar os pesos e as bias

            perda_acumulada += perda_instantanea.item() # Atualização da perda acumulada

        else:
            print("Epoch {} - Perda resultante: {}".format(epoch+1, perda_acumulada/len(trainloader)))
    print("\nTempo de treino (em minutos) =", (time()-inicio)/60)

def validacao(modelo, valloader, device):
    conta_corretas, conta_todas = 0, 0
    for imagens,etiquetas in valloader:
        for i in range(len(etiquetas)):
            img = imagens[i].view(1, 784)
            # Desativar o autograd para acelerar a validação
            with torch.no_grad():
                logps = modelo(img.to(device)) # Output do modelo em escala


            ps = torch.exp(logps) # Converte output para escala normal
            probab = list(ps.cpu().numpy()[0])
            etiqueta_pred = probab.index(max(probab)) # Converte o tensor em número
            etiqueta_certa = etiquetas.numpy()[i]
            if (etiqueta_certa == etiqueta_pred): # Compara a previsão com o valor correto
                conta_corretas += 1
            conta_todas += 1

    print("Total de imagens testadas =", conta_todas)
    print("\nPrecisão do modelo = {}%".format(conta_corretas*100/conta_todas))

modelo = Modelo() # Inicializa o modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Modelo rodará na GPU se possível
modelo.to(device)

print(modelo)  # Exibe a arquitetura da rede
treino(modelo, trainloader, device)
validacao(modelo, valloader, device)
