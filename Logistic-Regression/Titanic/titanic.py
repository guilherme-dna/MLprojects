import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Carregar o arquivo CSV em um DataFrame pandas
df = pd.read_csv('data/train.csv')
media_idade = df['Age'].mean()

# Preencher os valores ausentes na coluna 'Age' com a média da idade
df['Age'] = df['Age'].fillna(media_idade)

#removendo coluna inutil
df = df.drop("PassengerId",axis=1)

#pegando coluna target
target = df["Survived"]
target = target.to_numpy()


#queremos classificar dados de entrada em 2 grupos, resultado discreto (sim/nao , vive/morre , etc)

#sigmoide  = 1/ 1 + e^(b0 + b1x)   , x=[0,1]

#funcao de custo (aproximador dos pesos) = cross-entropy loss


#Escolhe aqui as variaveis a serem usadas pro modelo
variaveis=['Pclass','Sex','Age','SibSp','Parch','Fare']
df = df[variaveis]




# Converter o DataFrame em um array numpy
data = df.to_numpy()


#mapeando genero (male,female) pra (0,1)
if 'Sex' in variaveis:
    col_idx = variaveis.index('Sex')  # encontra em qual coluna está 'Sex'
    data[:, col_idx] = [1 if x == 'male' else 0 for x in data[:, col_idx]]



beta0= 2  #intercepto (valor base q entra no calculo de z)


#pesos do modelo
pesos=np.ones(len(variaveis)) 


N=len(data[:,0])


#taxa de passo dos pesos
taxa_aprendizado = 0.002    


#valor minimo de um passo para encerrar o treinamento
precisao=0.1


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def error(pesos):
    res = np.zeros_like(pesos)
    for i in range(N):
        z = beta0
        for j in range (len(variaveis)):
            z += pesos[j] * data[i, j]
        y_chapeu = sigmoide(z)
        erro = -1 / N * (target[i] - y_chapeu)

        for j in range (len(variaveis)):
            res[j] += erro * data[i, j]
    return res


inicio = time.time()


#calculamos o primeiro erro de tds variaveis
gradiente=error(pesos)
pesos -= taxa_aprendizado * gradiente


#Loop de treinamento
while np.sum(np.abs(gradiente))>precisao:
    gradiente=error(pesos)
    pesos -= taxa_aprendizado * gradiente
    



fim = time.time()

# Calcula o tempo de execução
tempo_execucao = fim - inicio


'''
#-----------------------------------------------------------------------plotando --------------------------------------------------------------------------

for var in variaveis:
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, ref_data[var]], data[:, ref_data['survive']], color='blue', label='Data')
    plt.xlabel(var)
    plt.ylabel('Sobrevivência')
    x_values = np.linspace(min(data[:, ref_data[var]]), max(data[:, ref_data[var]]), 100)
    z_values = beta0 + pesos[ref_data[var]] * x_values
    y_values = sigmoide(z_values)
    plt.plot(x_values, y_values, color='red', label='Sigmoide')
    plt.legend()
    plt.title(f'Sobrevivência vs {var}')
    plt.show()


#------------------------------------------------------testando dados apos treinamento --------------------------------------------------------------------
'''

# -------------------- TESTE --------------------

df_novos_dados = pd.read_csv('data/test.csv')

# salvar PassengerId antes de remover colunas
passenger_ids = df_novos_dados["PassengerId"]

# tratar idade
media_idade = df_novos_dados['Age'].mean()
df_novos_dados['Age'] = df_novos_dados['Age'].fillna(media_idade)

# mapear sexo
mapeamento = {'male': 1, 'female': 0}
df_novos_dados['Sex'] = df_novos_dados['Sex'].map(mapeamento)

# manter apenas as variáveis usadas no treino
df_novos_dados = df_novos_dados[variaveis]

# converter para numpy e garantir float
X_test = df_novos_dados.to_numpy().astype(float)

# calcular predições (vetorizado)
z = beta0 + np.dot(X_test, pesos)
y_hat = sigmoide(z)

previsoes = (y_hat > 0.5).astype(int)

# criar dataframe final
df_resultado = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": previsoes
})

# salvar csv
df_resultado.to_csv('resultado_previsto_tudo.csv', index=False)

print("Tempo de execução:", tempo_execucao, "segundos")