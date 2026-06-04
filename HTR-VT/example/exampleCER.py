import os
import re
import sys
import argparse
import editdistance
from collections import OrderedDict

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

# Agrega la carpeta principal del proyecto a las rutas del sistema para que Python pueda encontrar e importar los modulos utils, dataset y model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
from data import dataset
from model import HTR_VT

# Prepara una imagen cruda manteniendo su proporcion real y rellenando con blanco
def preprocess_image(image_path, max_w=512, max_h=64):
    # Abre la imagen en escala de grises
    image = Image.open(image_path).convert('L')
    img_w, img_h = image.size
    
    # Calcula el nuevo ancho manteniendo la proporcion intacta (regla de tres)
    new_w = min(int(img_w * max_h / img_h), max_w)
    
    # Redimensiona la imagen sin deformarla
    image = image.resize((new_w, max_h), Image.Resampling.BILINEAR)
    
    # Crea un lienzo nuevo completamente blanco (255) del tamaño exacto que exige la red (512x64)
    new_image = Image.new('L', (max_w, max_h), 255)
    
    # Pega la palabra redimensionada en el lado izquierdo del lienzo blanco
    new_image.paste(image, (0, 0))
    
    # Convierte a tensor y añade la dimension del batch
    transform_fn = transforms.ToTensor()
    image_tensor = transform_fn(new_image).unsqueeze(0)
    
    return image_tensor

# Ensambla la red, lee la imagen, predice el texto y lo compara con el texto real para que te devuelva el CER exacto
def main():
    # Inicia el analizador de argumentos para que puedas pasar la imagen y el texto real desde la terminal
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_cls', type=int, default=80)
    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+')
    parser.add_argument('--data_path', type=str, default='data/iam/lines/')
    parser.add_argument('--pth_path', type=str, default='output/iam/best_CER.pth')
    parser.add_argument('--train_data_list', type=str, default='data/iam/train.ln')
    parser.add_argument('--seed', type=int, default=1234)
    
    # DTROCR
    # Argumentos clave para que indiques que foto analizar y que texto usar como referencia
    # C:/Users/Usuario/Downloads/PFC1/DTrOCR/iam_words/words/a01/a01-000u/a01-000u-01-05.png    Peers
    # C:/Users/Usuario/Downloads/PFC1/DTrOCR/iam_words/words/a01/a01-000u/a01-000u-00-00.png A
    # C:/Users/Usuario/Downloads/PFC1/DTrOCR/iam_words/words/a01/a01-000u/a01-000u-00-01.png move
    
    parser.add_argument('--image_path', type=str, default='C:/Users/Usuario/Downloads/PFC1/DTrOCR/iam_words/words/a01/a01-000u/a01-000u-00-00.png', help='Ruta de la imagen a analizar')
    parser.add_argument('--texto_real', type=str, default='A', help='Texto escrito por humano para el CER')

    # HTR-VT
    # C:/Users/Usuario/Downloads/PFC1/HTR-VT/data/iam/lines/a01-000u-00.png     A MOVE to stop Mr. Gaitskell from
    # C:/Users/Usuario/Downloads/PFC1/HTR-VT/data/iam/lines/p03-185-01.png     Diana was trailing up the gravelled drive to the hospital
    
    # parser.add_argument('--image_path', type=str, default='C:/Users/Usuario/Downloads/PFC1/HTR-VT/data/iam/lines/p03-185-01.png ', help='Ruta de la imagen a analizar')
    # parser.add_argument('--texto_real', type=str, default='Diana was trailing up the gravelled drive to the hospital', help='Texto escrito por humano para el CER')

    
    args = parser.parse_args()

    # Asigna la tarjeta grafica (cuda) o el procesador (cpu) para que los calculos se hagan donde haya recursos
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Fabrica el esqueleto vacio de la red HTR-VT para que este listo para recibir los pesos
    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
    
    # Carga el archivo .pth a la memoria para que podamos extraer sus conocimientos
    ckpt = torch.load(args.pth_path, map_location='cpu')

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    
    # Itera sobre la red sombra (EMA) para que usemos la version mas inteligente del modelo
    for k, v in ckpt['state_dict_ema'].items():
        if re.search(pattern, k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    # Inyecta los pesos limpios en la red para que adquiera la capacidad de leer
    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    # Congela el modelo (eval) para que tecnicas como el dropout no alteren tu resultado final
    model.eval()

    # Carga metadatos del dataset para que extraiga el abecedario (ralph) exacto del entrenamiento
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    # Procesa la foto fisica para que se vuelva un tensor compatible
    image_tensor = preprocess_image(args.image_path).to(device)

    # Bloquea los gradientes (no_grad) para que la GPU ahorre memoria y solo infiera
    with torch.no_grad():
        # Genera las predicciones crudas
        preds = model(image_tensor)
        preds = preds.float()
        preds_size = torch.IntTensor([preds.size(1)])
        preds = preds.permute(1, 0, 2).log_softmax(2)
        
        # Elige los caracteres con mayor probabilidad para que armen la secuencia ganadora
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        
        # Traduce los numeros de vuelta a letras para que los humanos lo puedan leer
        preds_str = converter.decode(preds_index.data, preds_size.data)
        texto_predicho = preds_str[0]

    # ASIGNACION DE VARIABLES FINALES
    texto_real = args.texto_real
    ruta_imagen = args.image_path
    
    # Usa editdistance para que calcule cuantas inserciones, borrados o reemplazos hay entre la prediccion y la realidad
    distancia = editdistance.eval(texto_predicho, texto_real)
    
    # Protege contra division por cero en caso de que accidentalmente pases un texto real vacio para que no se caiga el programa
    if len(texto_real) == 0:
        error_rate = 1.0 if len(texto_predicho) > 0 else 0.0
    else:
        # Divide los fallos entre el total de letras reales para que obtengas la proporcion del error (CER)
        error_rate = distancia / float(len(texto_real))

    # IMPRESION EN EL FORMATO ESTRICTO SOLICITADO
    print("\n======================================")
    print(f"Ruta de imagen : {ruta_imagen}")
    print(f"Texto Real     : '{texto_real}'")
    print(f"Texto Predicho : '{texto_predicho}'")
    print(f"CER de la imagen: {error_rate * 100:.2f}%")
    print("======================================")


if __name__ == '__main__':
    main()