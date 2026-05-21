import os
import re
import sys
import argparse
from collections import OrderedDict

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from data import dataset
from model import HTR_VT


# Prepara una imagen para que tenga el formato, tamaño y colores que el modelo Transformer procese
def preprocess_image(image_path):
    # abre la imagen desde el disco duro y se convierte al modo 'L' (escala de grises) para que el modelo procese solo formas y contraste, descartando informacion de color inutil que lo confundiria
    image = Image.open(image_path).convert('L')
    
    # crea una tuberia de transformaciones (Compose) para que la imagen pase por varios filtros de estandarizacion 
    transform_fn = transforms.Compose([
        # se fuerza el tamaño a exactamente 64 pixeles de alto por 512 de ancho para que coincida milimetricamente con la cuadrícula de entrada que espera el codificador
        transforms.Resize(tuple([64, 512])),
        # convierte la imagen visual en una matriz y se normalizan sus valores entre 0 y 1 para que las neuronas puedan calcularla
        transforms.ToTensor()
    ])
    
    # Se añade una dimension extra al inicio con unsqueeze(0) para que la imagen aparente ser un (batch) de tamaño 1, ya que la red neuronal  exige recibir lotes de imagenes
    image_tensor = transform_fn(image).unsqueeze(0)
    
    # Se devuelve el tensor tridimensional formateado para que sea inyectado en el modelo
    return image_tensor

# ensambla la red, le coloca el modelo entrenado, toma una imagen, la hace pasar por la red y traduce la salida a texto 
def main():
    parser = argparse.ArgumentParser()
    # Se definen los argumentos de arquitectura y rutas para que sepa como construir el modelo y donde ir a buscar los pesos entrenados y la imagen
    parser.add_argument('--nb_cls', type=int, default=90)
    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+')
    parser.add_argument('--data_path', type=str, default='../data/iam/lines/')
    parser.add_argument('--pth_path', type=str, default='../data/iam/best_CER.pth')
    parser.add_argument('--train_data_list', type=str, default='../data/iam/train.ln')
    parser.add_argument('--seed', type=int, default=1234)
    # Ruta por defecto de la imagen que se quiere probar
    parser.add_argument('--image_path', type=str, default='./valid_1.jpeg')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # fabrica el modelo de la red HTR-VT para que recibiba la informacion aprendida
    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
    
    # Se carga el archivo de pesos (.pth) a la memoria RAM para que extraer sus pesos
    ckpt = torch.load(args.pth_path, map_location='cpu')

    # Se crea un diccionario ordenado vacio para que almacenemos los pesos
    model_dict = OrderedDict()
    # Se crea una expresion regular para buscar la palabra 'module.' en los nombres de las capas
    pattern = re.compile('module.')
    
    # Se itera sobre la copia EMA guardada en el archivo para que usemos la mejor version del modelo
    for k, v in ckpt['state_dict_ema'].items():
        if re.search(pattern, k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    # inyectan los pesos en la red 
    model.load_state_dict(model_dict, strict=True)
    # manda el modelo ensamblado a la memoria
    model = model.to(device)
    # apaga el modo de entrenamiento (eval) para que la red no use tecnicas de regularizacion como dropout que podrian alterar la prediccion 
    model.eval()

    # cargar el dataset de entrenamiento (metadatos) para que obtenga el alfabeto (ralph) que el modelo uso para aprender
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    # le pasa el alfabeto al convertidor CTC para que construya el diccionario traductor de numeros a letras
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    # llama a la funcion con la ruta de la iamgen para que la devuelva como un tensor del formato correcto
    image_tensor = preprocess_image(args.image_path)
    # Se envia la foto preparada a la tarjeta grafica para que se una al modelo
    image_tensor = image_tensor.to(device)

    # Se bloquea el calculo matematico de derivadas (no_grad) para que la GPU solo procese la imagen hacia adelante
    with torch.no_grad():
        # introduce la foto en el modelo para que genere los mapas densos de prediccion
        preds = model(image_tensor)
        # asegura que la matriz este en formato  para que conserve todos sus decimales de probabilidad
        preds = preds.float()
        # calcula el tamaño horizontal de la prediccion para que el decodificador CTC sepa cuantas columnas tiene que analizar
        preds_size = torch.IntTensor([preds.size(1)])
        # reordenan las dimensiones y se aplica log_softmax para que los numeros se conviertan en probabilidades del caracter
        preds = preds.permute(1, 0, 2).log_softmax(2)
        
        # busca el indice con la probabilidad mas alta (max) en el eje del vocabulario para que el sistema elija una sola letra ganadora por cada posicion de la imagen
        _, preds_index = preds.max(2)
        # aplana el resultado en un vector unidimensional lineal 
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        
        # le pasa la secuencia numerica ganadora al traductor CTC para que colapse los caracteres repetidos, elimine los tokens vacios, y devuelva la cadena de texto 
        preds_str = converter.decode(preds_index.data, preds_size.data)
        
        # extrae el primer elemento de la lista devuelta (ya que solo le enviamos una imagen) para que quede como una variable de texto
        recognized_text = preds_str[0]

    # imprime el resultado final
    print(f"Recognized_text: {recognized_text}")

if __name__ == '__main__':
    main()