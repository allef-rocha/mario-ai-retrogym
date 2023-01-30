import numpy as np
from itertools import product

sprites_ignore = [0x74, 0x78, 0x3E, 0x8E, 0x7B, 0x21]
sprites_ground = [0x83, 0x84, 0xB9]

'''
Extração de atributos da memória RAM do jogo Super Mario World
Informações retiradas de: https://www.smwcentral.net/?p=nmap&m=smwram
'''
def getXY(ram):
    '''
    getXY(ram): retorna informação da posição do agente
    embora layer1? não seja utilizada no momento, pode ser útil em algumas 
    alterações do algoritmo de aprendizado.
    '''
    
    # Coordenadas x, y em relação a fase inteira 
    # Elas estão armazenadas em 2 bytes cada
    # no formato little endian
    marioX = ram[0x95]*256 + ram[0x94]
    marioY = ram[0x97]*256 + ram[0x96]
    
    # Coordenada da parte visível do site
    layer1x = ram[0x1B]*256 + ram[0x1A]
    layer1y = ram[0x1D]*256 + ram[0x1C]
    
    return marioX.astype(np.int16), marioY.astype(np.int16), layer1x.astype(np.int16), layer1y.astype(np.int16)

def getSprites(ram):
  '''
  getSprites(ram): retorna os sprites (blocos, inimigos, itens) exibidos na tela.
  '''
  
  sprites = []
  
  # Podem existir até 12 sprites na tela
  for slot in range(12):
    # se o status é 0, não tem sprite nesse slot
    status = ram[0x14C8+slot]
    if status != 0:
      # posição x,y do sprite
      spriteX    = ram[0xE4+slot] + ram[0x14E0+slot]*256
      spriteY    = ram[0xD8+slot] + ram[0x14D4+slot]*256
      
      spriteSize = ram[0x0420+ram[0x15EA+slot]]  # tamanho do sprite
      spriteId   = ram[0x15EA+slot]              # qual é o sprite?
      spriteLabel = int(ram[0x9E + slot])  # https://smwspeedruns.com/Sprites
      
      # se for item (44) ou sprites ignoráveis, não insere
      if spriteId != 44 and spriteLabel not in sprites_ignore:
        # ou ele é 1x1 ou 4x4 blocos da nossa janela
        size = 1
        if spriteSize == 0:
          size = 4
        is_ground = spriteLabel in sprites_ground
        sprites.append({'x': spriteX, 'y': spriteY, 'size': size, 'ground':is_ground})
      
  return sprites
      
def getTile(dx,dy,ram):
  '''
  getTile(dx, dy, ram): retorna se tem um bloco que o mario possa pisar na posição dx, dy
  '''
  x = np.floor(dx/16)
  y = np.floor(dy/16)
  
  # 0x1C800 indica para cada pixel se é um obstáculo ou não
  # como obter o ponto certo foi retirado daqui: https://www.smwcentral.net/?p=viewthread&t=78887
  # return ram[0x1C800 + np.int(np.floor(x/16)*432 + y*16 + x%16)]

  # O endereço correto é 0x1F000, contribuição de Fernando Teixeira
  return ram[0x1F000 + np.int(np.floor(x/16)*432 + y*16 + x%16)]
  
def getInputs(ram, radius=6):
  '''
  getInputs(ram): retorna uma nd.array de inimigos, obstáculos dentro de um raio em torno do agente
  '''
  
  marioX, marioY, layer1x, layer1y = getXY(ram)
  sprites = getSprites(ram)

  # tamanho do vetor
  maxlen = (radius*2+1)*(radius*2+1)
  inputs = np.zeros(maxlen, dtype=int)
  
  # cada bloco de imagem representa 16x16 pixesl
  # portanto tendo um x,y de referência do Mario
  # devemos camihar de 16 em 16
  window = (-radius*16, radius*16 + 1, 16)
  j = 0

  def withinLimits(idx, ds1, ds2, r, maxlen):
    return (idx%(2*r + 1) + ds2 < 2*r + 1) and (idx + ds1*(2*r + 1) + ds2 < maxlen)
  
  for dy, dx in product(range(*window), repeat=2):
    # verifica se tem obstáculo na posição x+dx, y+dy
    # o +8 é para começar a medir a partir do meio do Mario
    tile = getTile(marioX+dx+8, marioY+dy, ram)
    
    # O Mario está sempre no meio, 
    # deve checar se o y está dentro do limite
    if tile==1 and marioY+dy < 0x1B0:
      inputs[j] = 1
    
    # Para cada sprite  
    for i, sprite in enumerate(sprites):
      # Se estiver dentro do bloco de 16 x 16 (-8, +8)
      distx = np.abs(sprite['x'] - marioX - dx)
      disty = np.abs(sprite['y'] - marioY - dy)
      size = sprite['size']
      if distx <= 8 and disty <= 8:
        # se estiver dentro dos limites, insira -1
        for s1, s2 in product(range(size), repeat=2):
          if withinLimits(j, s1, s2, radius, maxlen):
            # verifica se é 'chão' ou inimigo
            inputs[j + s1*(radius*2 + 1) + s2] = 1 if sprite['ground'] else -1
    j = j + 1
  return inputs, marioX, marioY

# Recupera o estado atual como um array bidimensional
def getState(ram, radius):
  state, x, y = getInputs(ram, radius=radius)
  # rstate = np.reshape(state, (2*radius + 1, 2*radius + 1))
  
  return state, x, y
  

def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)
    