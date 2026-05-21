import torch
import torch.distributed as dist
from torch.distributions.uniform import Uniform

import os
import re
import sys
import math
import logging
from copy import deepcopy
from collections import OrderedDict

# variable global evaluando si hay una tarjeta grafica disponible ('cuda') para que asigne automaticamente los calculos matriciales pesados a la GPU, o al procesador ('cpu') si no la hay.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Genera un numero entero aleatorio
def randint(low, high):
    # genera un tensor de un solo valor con torch.randint entre los limites y se extrae en int() que las funciones de distorsion puedan usar
    return int(torch.randint(low, high, (1, )))


# numero decimal aleatorio.
def rand_uniform(low, high):
    # Uniform para muestrear un numero aleatorio continuo en float para probabilidades
    return float(Uniform(low, high).sample())


# Crea el log que imprime los mensajes en la consola y al mismo tiempo los guarda en un archivo .log.
def get_logger(out_dir):
    # inicializa un objeto Logger llamado 'Exp' para que centralice todos los mensajes del experimento
    logger = logging.getLogger('Exp')
    # establece el nivel en INFO para que ignore mensajes de depuracion menores y solo guarde lo importante
    logger.setLevel(logging.INFO)
    # crea la plantilla 'formatter' para que cada mensaje indique automaticamente la hora exacta (asctime), el nivel de texto
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # une la ruta de salida con el nombre 'run.log'
    file_path = os.path.join(out_dir, "run.log")
    # para que se encargue de escribir los datos 
    file_hdlr = logging.FileHandler(file_path)
    # asigna la plantilla de formato al escritor de archivo
    file_hdlr.setFormatter(formatter)

    # StreamHandler para que, simultaneamente, de los mismos mensajes en la pantalla de la consola (sys.stdout)
    strm_hdlr = logging.StreamHandler(sys.stdout)
    #asigna la plantilla de formato para la consola
    strm_hdlr.setFormatter(formatter)

    # conectan ambos handlers al logger para que cuando el codigo use logger.info(), se active la escritura en archivo y en pantalla 
    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


# Actualiza la Tasa de Aprendizaje (Learning Rate) durante el entrenamiento, subiendola al inicio y bajandola progresivamente imitando la curva de una onda coseno
def update_lr_cos(nb_iter, warm_up_iter, total_iter, max_lr, optimizer, min_lr=1e-7):
    # evalua si la iteracion actual es menor a la fase de calentamiento para que la red no de pasos muy grandes al principio y evite colapsar
    if nb_iter < warm_up_iter:
        # calcula la tasa actual multiplicando el maximo por una fraccion lineal para que suba escalonadamente hasta llegar al limite maximo
        current_lr = max_lr * (nb_iter + 1) / (warm_up_iter + 1)
    else:
        # calcula la tasa usando la funcion math.cos para que decaiga, permitiendo que la red aterrice en el valle de error sin omitir el minimo
        current_lr = min_lr + (max_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * nb_iter / (total_iter - warm_up_iter)))

    # recorren los grupos de parametros del optimizador para que se inyecte directamente este valor calculado en el modelo
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    # devuelve el optimizador alterado y la tasa actual para que train.py los utilice y los imprima
    return optimizer, current_lr


# diccionario para letras a identificadores numericos para que la red calcule el error, y luego traduce los numeros devuelta a letras para que  leer las predicciones
class CTCLabelConverter(object):
    def __init__(self, character):
        # obtiene los caracteres del abecedario para que queden en una lista
        dict_character = list(character)
        self.dict = {}
        # bucle para que asigne un numero unico (i + 1) a cada letra, desde 1
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
            
        # evalua si hay exactamente 87 caracteres para que, si es el dataset IAM, añada los corchetes faltantes manualmente y evite que la red explote al verlos en el conjunto de validacion
        if len(self.dict) == 87:     
            self.dict['['], self.dict[']'] = 88, 89
            
        # añade un token especial '[blank]' al indice 0 para que la funcion CTC lo use como "espacio vacio", permitiendole saber cuando termina de dibujar una letra repetida
        self.character = ['[blank]'] + dict_character

    def encode(self, text):
        # calcula la cantidad de letras de cada palabra para que la funcion de perdida CTC sepa cuanto texto debe esperar en cada imagen
        length = [len(s) for s in text]
        
        # unen todos los textos en una sola cadena para que sea mas rapido procesarlos todos juntos
        text = ''.join(text)
        # traduce cada caracter 'char' usando 'self.dict' para que se conviertan en una gran lista de numeros enteros
        text = [self.dict[char] for char in text]

        # envuelven las listas numericas en 'torch.IntTensor' y se envian a la tarjeta grafica (to(device)) para que esten listos para la formula CTC
        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        index = 0

        # itera sobre las longitudes predichas para que el codigo sepa donde cortar el bloque de predicciones y separar las palabras individuales
        for l in length:
            # extrae el segmento exacto de numeros correspondientes a una sola imagen
            t = text_index[index:index + l]
            char_list = []
            
            # itera sobre los numeros de este segmento para que sean traducidos uno por uno
            for i in range(l):
                # evalua que el numero no sea 0 (token blanco), que no sea exactamente igual a la letra anterior (para colapsar letras repetidas por la CTC) y que no sobrepase el limite del diccionario
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i]<len(self.character):
                    # busca la letra equivalente en 'self.character' y se añade a la lista para que se construya la palabra final
                    char_list.append(self.character[t[i]])
                    
            # juntan todas las letras decodificadas sin espacios para que formen la palabra predicha 
            text = ''.join(char_list)

            # añade a la lista de respuestas totales y se actualiza el indice para saltar a la siguiente imagen
            texts.append(text)
            index += l
            
        # devuelven las cadenas de texto legibles por humanos para que impriman el resultado final
        return texts


#  promedios. Acumula los errores de cada iteracion y al final da el promedio total 
class Averager(object):
    def __init__(self):
        # Llama a la funcion reset para que los contadores empiecen en cero
        self.reset()

    def add(self, v):
        # extrae la cantidad de elementos en el tensor para que sepa cuantos datos nuevos estan entrando
        count = v.data.numel()
        # suman todos los valores del tensor para que quede un solo numero compacto
        v = v.data.sum()
        # suman ambas cifras a los acumuladores globales para que mantengan el total historico intacto
        self.n_count += count
        self.sum += v

    def reset(self):
        # igualan las variables a cero para que se pueda calcular un nuevo promedio en la siguiente epoca sin arrastrar datos viejos
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        # previene la division por cero evaluando que n_count tenga datos 
        if self.n_count != 0:
            # divide la suma total entre el conteo para que retorne el promedio matematico final
            res = self.sum / float(self.n_count)
        return res


# Similar a Averager, pero para el modo multi GPU. Comunica los errores entre diferentes tarjetas de video
class Metric(object):
    def __init__(self, name=''):
        self.name = name
        self.sum = torch.tensor(0.).double()
        self.n = torch.tensor(0.)

    def update(self, val):
        rt = val.clone()
        # usa 'dist.all_reduce' para que recoja los valores de esta metrica de TODAS las tarjetas de video instaladas y los sume en la variable rt
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        # divide entre el total de tarjetas usadas (world_size) para que el error no este inflado artificialmente
        rt /= dist.get_world_size()
        # acumulan los valores en el procesador principal (cpu) para que no ocupen espacio en la tarjeta grafica
        self.sum += rt.detach().cpu().double()
        self.n += 1

    @property
    def avg(self):
        # retorna la division para que entregue el promedio sincronizado de el cluster de GPUs
        return self.sum / self.n.double()


# ModelEma (Exponential Moving Average)
# Crea una copia de la red. Mientras la red principal actualiza sus pesos, esta red actualiza los suyos sacando un promedio de los pesos anteriores, esto hace que las predicciones finales sean mas estables
class ModelEma:
    def __init__(self, model, decay=0.9999, device='', resume=''):
        # hace una copia fisicamente exacta e independiente (deepcopy) del modelo principal 
        self.ema = deepcopy(model)
        # apaga el modo de entrenamiento en la copia con eval() para que no sufra alteraciones por Dropout o BatchNorm durante la corrida
        self.ema.eval()
        # guarda la tasa de suavizado (decay) para que la clase sepa que tanto peso darle a los valores viejos frente a los valores nuevos
        self.decay = decay
        self.device = device
        # envia la red copia a la tarjeta de video para que las operaciones matriciales de promediado sean rapidas
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        
        # se pasa una ruta de guardado, se carga para que el promedio continue desde donde se pauso el entrenamiento 
        if resume:
            self._load_checkpoint(resume)
            
        # recorren los parametros y se apaga el requerimiento de gradientes para que PyTorch sepa que no debe perder memoria intentando calcular derivadas sobre esta copia
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path, mapl=None):
        # carga el archivo fisico (.pth) para que se extraiga el diccionario interno de pesos viejos
        checkpoint = torch.load(checkpoint_path,map_location=mapl)
        assert isinstance(checkpoint, dict)
        
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            # evalua y formatea el nombre de cada capa para que coincida perfectamente con el formato interno de PyTorch, quitando o poniendo el prefijo 'module.' segun sea necesario
            for k, v in checkpoint['state_dict_ema'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            # sobreescriben los pesos de la copia con los pesos cargados del archivo para que este lista para operar
            self.ema.load_state_dict(new_state_dict)
            print("=> Loaded state_dict_ema")
        else:
            print("=> Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model, num_updates=-1):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        
        # ajusta la tasa de decaimiento en las primeras iteraciones para que la copia asimile los cambios al principio
        if num_updates >= 0:
            _cdecay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        else:
            _cdecay = self.decay

        # envuelve el codigo en no_grad() para que se bloquee estrictamente la construccion de grafos computacionales matematicos ahorrando RAM
        with torch.no_grad():
            msd = model.state_dict()
            # recorre cada peso individual de la sombra
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                    
                # extrae el peso equivalente de la red principal y se desprende de su arbol de operaciones (detach) para que sea un numero puro
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                    
                # sobreescribe el peso de la sombra usando la formula de Promedio Exponencial, multiplicando lo viejo por el decay (99.99%) y lo nuevo por el remanente (0.01%) 
                ema_v.copy_(ema_v * _cdecay + (1. - _cdecay) * model_v)


# prepara los textos separando los simbolos para que el calculo de Error de Palabras (WER) sea justo y no marque como fallo una palabra entera solo porque tenia un punto pegado al final
def format_string_for_wer(str):
    # Se usa expresiones regulares (Regex) para que intercepte cualquier signo de puntuacion en la cadena y le agregue espacios artificiales a su izquierda y derecha
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    # limpia la cadena aplastando multiples espacios repetidos seguidos en un solo espacio limpio (strip) para que al momento de dividir las palabras no se creen tokens vacios que arruinen el conteo final
    str = re.sub('([ \n])+', " ", str).strip()
    return str