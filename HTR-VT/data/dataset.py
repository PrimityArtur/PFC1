import numpy as np
import torch
import skimage
import os
import itertools
from PIL import Image
from torch.utils.data import Dataset
from utils import utils
from data import transform as transform
from torchvision.transforms import ColorJitter


# Se utiliza como empaquetador final antes de enviar los datos a la GPU
# Recibe un lote batch de imagenes y textos, y les aplica las deformaciones de transform 
# de forma aleatorias y los convierte en tensores matematicos para que la red no de error.
def SameTrCollate(batch, args):
    # Se usa zip(*batch) sobre la tupla para separar todas las imagenes a un lado y todas las etiquetas de texto al otro
    images, labels = zip(*batch)
    
    # Se reconstruyen las matrices numericas a formato visual Image de la libreria PIL, multiplicando por 255, para que las funciones de transformacion visual puedan manipular
    images = [Image.fromarray(np.uint8(images[i][0] * 255)) for i in range(len(images))]

    # Se genera un aleatorio menor a 0.5 para que haya un 50% de probabilidad de inclinar las imagenes, y tener caligrafias en diagonal
    if np.random.rand() < 0.5:
        images = [transform.RandomTransform(args.proj)(image) for image in images]

    # aca lo mismo se decide si aplicar variaciones de grosor al trazo de la tinta
    if np.random.rand() < 0.5:
        # Se genera un tamaño de kernel aleatorio para que el engrosamiento o adelgazamiento sea impredecible y no siempre igual
        kernel_h = utils.randint(1, args.dila_ero_max_kernel + 1)
        kernel_w = utils.randint(1, args.dila_ero_max_kernel + 1)
        
        # Se aplica Erosion para que los trazos de las letras se vuelvan muy delgados y se simulen escrituras finas
        # Dilation para que los trazos se ensanchen simulando tinta engrosadas
        if utils.randint(0, 2) == 0:
            images = [transform.Erosion((kernel_w, kernel_h), args.dila_ero_iter)(image) for image in images]
        else:
            images = [transform.Dilation((kernel_w, kernel_h), args.dila_ero_iter)(image) for image in images]

    # aplica ColorJitter para alterar el brillo y el contraste, para tener imagenes oscuras o sobreexpuestas
    if np.random.rand() < 0.5:
        images = [ColorJitter(args.jitter_brightness, args.jitter_contrast, args.jitter_saturation,
                              args.jitter_hue)(image) for image in images]

    # Ahora convierte las imagenes deformadas de vuelta a matrices de PyTorch (tensors) para que puedan entrar al modelo
    image_tensors = [torch.from_numpy(np.array(image, copy=True)) for image in images]
    
    # aca usa torch.cat con un unsqueeze para apilar todas las matrices individuales en un solo bloque tridimensional, para que la GPU lo procese en paralelo
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
    
    # aca se añade una dimension extra (unsqueeze(1)) representando el canal de color (blanco y negro = 1) y se pasa a float para el formato que pide el codificador
    image_tensors = image_tensors.unsqueeze(1).float()
    
    # Se divide el tensor entre 255 para normalizar todos los pixeles entre 0.0 y 1.0, para que los pesos del optimizador converjan rapido y no se generen numeros infinitos
    image_tensors = image_tensors / 255.
    
    # Se devuelve el bloque de imagenes y sus textos correspondientes para poder entrenar
    return image_tensors, labels


# Ayuda a mantener un indice ordenado de todas las rutas de los archivos y buscar la imagen en memoria cuando se necesite
class myLoadDS(Dataset):
    def __init__(self, flist, dpath, img_size=[512, 32], ralph=None, fmin=True, mln=None):
        # lee el archivo train.ln y guarda en una lista todas las rutas de las imagenes
        self.fns = get_files(flist, dpath)
        
        # lee el contenido de los archivos .txt y tenga las respuestas cuando se solicite
        self.tlbls = get_labels(self.fns)
        self.img_size = img_size

        # Se verifica si ya nos dieron un alfabeto (cuando se evalua el test). Si no existe, se calcula uno nuevo
        if ralph == None:
            # genera el diccionario de caracteres unicos de todas las etiquetas para saber que letras se necesita predecir
            alph = get_alphabet(self.tlbls)
            # invierte el diccionario (para que la llave sea el numero y el valor la letra) para poder traducir la salida matematica a texto legible al final del proceso
            self.ralph = dict(zip(alph.values(), alph.keys()))
            self.alph = alph
        else:
            self.ralph = ralph

        # verifica si hay que filtrar por longitud maxima de caracteres
        if mln != None:
            # Se crea una mascara booleana para filtrar solo las oraciones que cumplan con la condicion de tamaño para evitar errores de (Out of Memory) en secuencias excesivamente largas
            filt = [len(x) <= mln if fmin else len(x) >= mln for x in self.tlbls]
            self.tlbls = np.asarray(self.tlbls)[filt].tolist()
            self.fns = np.asarray(self.fns)[filt].tolist()

    def __len__(self):
        # aca ya devuelve la cantidad total de imagenes disponibles para que PyTorch sepa cuando se termina una epoca de entrenamiento
        return len(self.fns)

    def __getitem__(self, index):
        # llama a la funcion get_images pasando el indice exacto para cargar solo la imagen a la memoria RAM
        timgs = get_images(self.fns[index], self.img_size[0], self.img_size[1])
        
        # cambia el orden de las dimensiones de (Alto, Ancho, Canales) a (Canales, Alto, Ancho) para el estandar que pide PyTorch en convoluciones
        timgs = timgs.transpose((2, 0, 1))

        # Se devuelve el par (imagen_preprocesada, texto_correcto) para que el DataLoader lo recoja y lo junte con otros para armar un lote
        return (timgs, self.tlbls[index])


def get_files(nfile, dpath):
    # Se leen todas las lineas del archivo de configuracion (por ejemplo train.ln) para extraer los nombres de los archivos
    fnames = open(nfile, 'r').readlines()
    # Se junta la ruta base (dpath) con cada nombre de archivo extraido limpiando saltos de linea (strip) para construir la ruta absoluta hacia la iamgen
    fnames = [dpath + x.strip() for x in fnames]
    return fnames

# Redimensiona una imagen manteniendo su proporcion original
def npThum(img, max_w, max_h):
    # extrae las dimensiones actuales de la imagen en las variables alto, ancho
    x, y = np.shape(img)[:2]

    # calcula la nueva longitud del ancho 'y' mediante regla de tres simple respecto a la nueva altura max_h, y se usa la funcion min para asegurar que no se desborde del max_w permitido
    y = min(int(y * max_h / x), max_w)
    
    # aca el alto 'x' sea el tamaño esperado max_h
    x = max_h

    # Se aplica el redimensionado a la imagen con la nueva resolucion calculada
    img = np.array(Image.fromarray(img).resize((y, x)))
    return img


# Carga la imagen y le añade un relleno de pixeles blancos a la derecha para que todas las imagenes del lote midan exactamente lo mismo
def get_images(fname, max_w=500, max_h=500, nch=1):
    try:
        # abre la imagen con PIL y se usa convert('L') para transformarla a escala de grises descartando color innecesario para simplificar el calculo de la red
        image_data = np.array(Image.open(fname).convert('L'))
        
        # npThum para achicar o agrandar la imagen hasta la altura esperada sin romper las proporciones del texto
        image_data = npThum(image_data, max_w, max_h)
        
        # Se convierte la matriz a tipo flotante de 32 bits para mas precision en los calculos matriciales de la NN
        image_data = skimage.img_as_float32(image_data)
       
        # Si la imagen no tiene dimension de canal (es plana), se le inyecta una dimension extra al final para evitar errores de forma
        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)

        # 3 canales (RGB) pero la imagen es blanco y negro, se usa np.tile para clonar el canal de grises 3 veces y simular un formato de color valido
        if nch == 3 and image_data.shape[2] != 3:
            image_data = np.tile(image_data, 3)

        # aplica np.pad para inyectar pixeles blancos (constant_values=1.0) al lado derecho de la imagen hasta alcanzar exactamente 'max_w' Esto se hace para que imagenes de palabras cortas y palabras largas puedan empaquetarse juntas en un tensor cuadriculado del mismo tamaño
        image_data = np.pad(image_data, ((0, 0), (0, max_w - np.shape(image_data)[1]), (0, 0)), mode='constant',
                            constant_values=(1.0))

    except IOError as e:
        print('Could not read:', fname, ':', e)

    return image_data


# Busca y limpia el texto correcto digitalizado para cada imagen
def get_labels(fnames):
    labels = []
    for id, image_file in enumerate(fnames):
        # Se usa os.path.splitext para borrar la extension '.png' y ponerle '.txt' para encontrar el archivo de transcripcion de esa misma imagen
        fn = os.path.splitext(image_file)[0] + '.txt'
        
        # lee el contenido de texto
        lbl = open(fn, 'r').read()
        
        # Se separan las palabras y se vuelven a unir con un solo espacio usando ' '.join(lbl.split()) para eliminar espacios dobles, tabulaciones y saltos de linea basura que no usa la funcion de perdida CTC.
        lbl = ' '.join(lbl.split())

        labels.append(lbl)

    return labels


# Analiza todos los textos del entrenamiento para crear el abecedario para la NN
def get_alphabet(labels):
    # concatenan todos los textos de entrenamiento 
    coll = ''.join(labels)
    
    # Se usa set() para descartar duplicados y extraer solo los caracteres unicos, y luego ordenarlos
    unq = sorted(list(set(coll)))
    
    # Se formatea la lista de caracteres unicos.
    unq = [''.join(i) for i in itertools.product(unq, repeat=1)]
    
    # Se crea un diccionario enlazando cada caracter unico a un numero entero (ej: 'a':0, 'b':1) para que el modelo predictivo asigne una neurona especifica a cada letra.
    alph = dict(zip(unq, range(len(unq))))

    return alph


# Son generadores infinitos (yield) que reinician el dataset. se usan para que el bucle del entrenamiento pueda contar hasta 100,000 iteraciones sin importar que se acaben las imagenes de una epoca volviendo a entregarlas 
def cycle_dpp(iterable):
    epoch = 0
    iterable.sampler.set_epoch(epoch)
    while True:
        for x in iterable:
            yield x
        epoch += 1
        iterable.sampler.set_epoch(epoch)


def cycle_data(iterable):
    while True:
        for x in iterable:
            yield x

