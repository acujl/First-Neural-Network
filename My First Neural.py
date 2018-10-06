from matplotlib import pyplot as plt
import numpy
from random import randint

# Definir os dados, [0] e [1] caracteristicas e [2] o tipo
dataB1 = [1, 1, 0]
dataB2 = [2, 1,   0]
dataB3 = [2, .5, 0]
dataB4 = [3,   1, 0]
dataR1 = [3, 1.5, 1]
dataR2 = [3.5,   .5, 1]
dataR3 = [4, 1.5, 1]
dataR4 = [5.5,   1,   1]

#Dado mistério
dataU = [4.5,  1, 1] #Deve ser 1 né
#TOTAL DADOS conhecidos
all_points = [dataB1, dataB2, dataB3, dataB4, dataR1, dataR2, dataR3, dataR4]

#--<@>--<@>--<@>--<@>--<@>--<@> VER DADOS <@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>
def vis_data():
    plt.grid()
    for i in range(len(all_points)):
        c = 'r'
        if all_points[i][2] == 0:
            c = 'b'
        plt.scatter([all_points[i][0]], [all_points[i][1]], c=c)
    plt.scatter([dataU[0]], [dataU[1]], c='gray')
    plt.show()
vis_data()
#--<@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>--<@>

#Definir funções
def sigmoid(x):
    return 1/(1 + numpy.exp(-x))
def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

#Começar training
def train():
    w1 = numpy.random.randn()*.2-.1
    w2 = numpy.random.randn()*.2-.1
    b = numpy.random.randn()*.2-.1
    learning_rate = 0.2
    interações=50000
    for i in range(interações):
        # escolher ponto random
        random_idx = numpy.random.randint(len(all_points))
        point = all_points[random_idx]
        target = point[2] # o nosso alvo

        # FEEEEEDDD
        z = w1 * point[0] + w2 * point[1] + b
        pred = sigmoid(z)

        # Comparamos com o target
        #  cost = (pred - target) ** 2

        # declive w.r.t. para cada parametro (w1, w2, b)
        # fazer a derivada atravez de ^2
        dcost_dpred = 2 * (pred - target)        #DERIVADA DO CUSTO
        dpred_dz = sigmoid(z) * (1 - sigmoid(z)) #DERIVADA DA SIGMOID

        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1

        # Regra da cadeira
        dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1
        dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2
        dcost_db =  dcost_dpred * dpred_dz * dz_db

        # atualizamos os parametros :D
        w1 -= learning_rate * dcost_dw1
        w2 -= learning_rate * dcost_dw2
        b -= learning_rate * dcost_db
    return w1, w2, b

w1, w2, b = train()

z = w1 * dataU[0] + w2 * dataU[1] + b
pred = sigmoid(z)

target = dataU[2]
resposta = round(pred)

if resposta == 1:
    print("Dado de tipo 1 (vermelho)")
else:
    print("Dado de tipo 0 (azul)")

cost = (pred - target) ** 2
confiança = 100 - cost
print("Confiança de: " + str(confiança) + "%")
train()

print("\nFeito por:\nwww.bilatech.net")





