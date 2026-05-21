import torch
import torch.nn as nn

# crea una capa de convolucion cada vez que se necesite escanear la imagen
def conv3x3(in_planes, out_planes, stride=1):
    # retorna una capa bidimensional para que escanee la imagen usando una ventana de 3x3 pixeles 
    # aplicando 'padding=1' para que agregue un borde invisible a la imagen y asi no se encoja despues de escanearla
    # Se usa 'bias=False' para ahorrar memoria, ya que la siguiente capa (BatchNorm) hara ese trabajo
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# base del ResNet q procesa la imagen para extraer detalles, pero al final le suma la imagen original. Esto se hace para que la red no olvide la informacion pasada 
class BasicBlock(nn.Module):
    # Se define la expansion en 1 para que la cantidad de canales de salida sea igual a la de entrada
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # la primera convolucion para que extraiga los primeros patrones caligraficos de este bloque
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        # normaliza por lotes (BatchNorm2d) para que estabilice y escale los valores matematicos resultantes de conv1, evitando que los numeros sean muy diferentes y acelerando el aprendizaje
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        
        # inicializa la funcion de activacion ReLU para que introduzca no-linealidad, permitiendo a la red aprender trazos curvos y formas complejas en lugar de solo lineas rectas
        self.relu = nn.ReLU(inplace=True)
        
        # segunda convolucion para que refine los patrones de la primera convolucion
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        
        # Se guarda la operacion de adaptacion 'downsample' para que la conexion residual pueda encogerse y encajar con la salida 
        self.downsample = downsample        
        self.stride = stride

    def forward(self, x):
        # guarda una copia de 'x' y represente la memoria a corto plazo
        residual = x

        # la imagen 'x' pasa por la primera convolucion para que extraiga las nuevas caracteristicas
        out = self.conv1(x)
        # normaliza el resultado
        out = self.bn1(out)
        # Se filtran los numeros negativos dejandolos en 0 con ReLU para que solo se activen las neuronas importantes
        out = self.relu(out)

        # Se pasa la imagen por la segunda convolucion
        out = self.conv2(out)
        out = self.bn2(out)

        # verifica si existe una funcion 'downsample' guardada para que se actue en caso de que la resolucion de la imagen haya cambiado
        if self.downsample is not None:
            # aplica el 'downsample' a la copia original 'residual' para que su tamaño matricial coincida exactamente con la nueva imagen 'out' y no de error al sumars
            residual = self.downsample(x)

        # suma la memoria intacta 'residual' con la informacion procesada 'out para que el modelo no pierda contexto.
        out += residual
        
        # y se pasa la suma final por ReLU para que la señal salga hacia el siguiente bloque de la red
        out = self.relu(out)

        return out


# reduce en mapas de caracteristicas (tokens visuales) para entregarselos al Transformer
class ResNet18(nn.Module):

    def __init__(self, nb_feat = 384):
        # Se define la cantidad inicial de canales dividiendo los atributos totales solicitados entre 4 (384 // 4 = 96) para que la red empiece el analisis visual con una carga de procesamiento ligero
        self.inplanes = nb_feat // 4
        
        super(ResNet18, self).__init__()
        
        # entrada de la red. Se usa kernel=3 y stride=(2,1) para que aplaste la imagen a la mitad de su altura pero conserve todo su ancho, para no distorsionar las palabras largas horizontales
        self.conv1 = nn.Conv2d(1, nb_feat // 4, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_feat // 4, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        
        # se crea una capa de MaxPool con stride rectangular (2,1) para que reduzca mas la altura de la imagen a la mitad, quedandose solo con los pixeles que tengan las lineas mas oscuras 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
        
        # capa 1 con 2 bloques residuales para que procesen la imagen inicial sin reducir mas su resolucion
        self.layer1 = self._make_layer(BasicBlock, nb_feat // 4, 2, stride=(2, 1))
        
        # capa 2 con 2 bloques mas, forzando un stride=2 para que el mapa visual se reduzca a la mitad en ambas direcciones, concentrando la informacion
        self.layer2 = self._make_layer(BasicBlock, nb_feat // 2, 2, stride=2)
        
        #capa 3, otros 2 bloques con compresion visual, duplicando los canales para representar texturas
        self.layer3 = self._make_layer(BasicBlock, nb_feat, 2, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        # se inicializa la variable de reduccion en Vacio (None) asumiendo por defecto que no habra cambios de tamaño en este bloque.
        downsample = None
        
        # evalua si hay un cambio en el salto de pixeles (stride) o en la cantidad de canales para que el algoritmo sepa que debe crear un adaptador de tamaño
        if stride != 1 or self.inplanes != planes * block.expansion:
            # empaqueta una convolucion pequeña (kernel=1) y un BatchNorm en la variable 'downsample' para redimensionar la matriz residual
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        # añade el primer bloque a la lista pasandole el adaptador 'downsample' para que solucione cualquier cambio de tamaño entre capas
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # se actualiza grosor de canales para que los siguientes bloques sepan cuanta informacion van a recibir
        self.inplanes = planes * block.expansion
        # bucle para generar los bloques restantes de la capa
        for i in range(1, blocks):
            # se añaden bloques (sin downsample) a la lista de capas para que profundicen el analisis visual en el nuevo tamaño ya estabilizado
            layers.append(block(self.inplanes, planes, 1, None))

        # se usa nn.Sequential para fusionar la lista de bloques en una sola 
        return nn.Sequential(*layers)

    def forward(self, x):
        # mapeo visual inicial
        x = self.conv1(x)
        # normaliza
        x = self.bn1(x)
        # descartar señales
        x = self.relu(x)
        # deseche el fondo inutil y achique el tamaño de la matriz 
        x = self.maxpool(x)

        # patrones simples como bordes
        x = self.layer1(x)
        # cruce la informacion y encuentre bucles o letras simples
        x = self.layer2(x)
        # genere representaciones de las palabras
        x = self.layer3(x)
        
        # ultimo filtrado MaxPool para que la matriz final quede concentrada con las caracteristicas semanticas mas relevantes
        x = self.maxpool(x)
        
        # (tokens visuales) para el Transformer las asimile y les de un contexto global
        return x