# Super Mario World - Inteligência Artificial

Prof. Fabrício Olivetti de França (folivetti@ufabc.edu.br)

## Instruções de instalação:

- Instale a biblioteca Retro Gym seguindo as instruções em: https://github.com/openai/retro
- Copie a ROM do jogo para o diretório *site-packages/retro/data/stable/SuperMarioWorldSnes/* com o nome *rom.sfc* (se estiver utilizando o Anaconda, ele deve
estar em *~/anaconda3/lib/python3.6/*)

## Requisitos

- python
- pip
- bibliotecas presentes no arquivo `requirements.txt`

Para realizar a instalação das bibliotecas, utilize o comando

`pip install -r requirements.txt`


## Execução

### Train

Para realizar o treinamento, utilize o comando

`python train.py -m`  

Para interromper a execução, a qualquer momento utilize o comando `ctrl` + `C`.  

Este programa tem suporte para as seguintes flags:

`-h`, `--help`: 
    Mostra mensagem de ajuda  

`-H`, `--hide`: 
    Oculta tela do jogo  

`-m`, `--multiprocess`: 
    Utilizar multiprocessamento (recomendado)  

`-s`, `--startover`: 
    Recomeça o treinamento do zero  

`-a`, `--agent [str]`: 
    Nome do arquivo que armazena o agente na pasta "agents/" (default: "current")

`-g`, `--generations [int]`: 
    Numero de gerações. 0 (zero) significa número indeterminado (default: 0)  

`-l`, `--level [str]`: 
    Level (default: "YoshiIsland2")  

`-p`, `--popsize [int]`: 
    Tamanho da população (default: 50)  

Exemplo:

`python train.py -m -l YoshiIsland1 -p 100`


### Play

Para mostrar o agente já treinado jogando a fase, utilize o comando

`python play.py`

Este programa tem suporte para as seguintes flags:

`-h`, `--help`: 
    Mostra mensagem de ajuda  

`-s`, `--show`: 
    Imprime o state no terminal  

`-a`, `--agent [str]`: 
    Nome do arquivo que armazena o agente na pasta "agents/" (default: "current)  

`-l`, `--level [str]`: 
    Level  (default: "YoshiIsland2")  

Exemplo:

`python play.py -s -a winner01`
