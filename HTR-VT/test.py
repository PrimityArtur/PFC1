import torch

import os
import re
import json
import valid
from utils import utils
from utils import option
from data import dataset
from model import HTR_VT
from collections import OrderedDict


# reconstruir la arquitectura de la red, inyectarle los pesos del entrenamiento (.pth) y probar imagenes nuevas, dondo los CER WER del papaer
def main():
    # evalua si hay una tarjeta grafica disponible ('cuda:0') o procesador normal ('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # fija la semilla 123 global para que, si se repite la prueba, las metricas sean iguales
    torch.manual_seed(args.seed)

    # reconstruye la ruta hacia la carpeta del experimento (output/iam) para buscar el modelo guardado
    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    
    # crea la carpeta si no existiera para el registro de logs 
    os.makedirs(args.save_dir, exist_ok=True)
    
    # activa el sistema de registro (logger) para que escriba los resultados de prueba 
    logger = utils.get_logger(args.save_dir)
    
    # imprimen todos los hiperparametros usados en la consola en formato JSON 
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # manda a construir el Transformer llamando a create_model para que recibiba los pesos
    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    # define la ruta exacta hacia el archivo que contiene el mejor modelo guardado durante el entrenamiento para que el script apunte al mayor logro de la red
    pth_path = args.save_dir + '/best_CER.pth'
    logger.info('loading HWR checkpoint from {}'.format(pth_path))

    # carga el archivo pesado de pesos (.pth) a la memoria RAM (cpu) para que pueda extraer sus diccionarios internos de parametros
    ckpt = torch.load(pth_path, map_location='cpu')
    
    # prepara un diccionario vacio ordenado (OrderedDict) y un patron de texto regex ('module.') para limpiar los nombres de las capas si es necesario
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    # itera sobre el diccionario de la copia EMA ('state_dict_ema') porque es la version mas estable del modelo para que se use en la prueba final
    for k, v in ckpt['state_dict_ema'].items():
        # busca si la capa empieza con la palabra 'module' (multiples GPUs) para que el codigo elimine ese prefijo
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            # Si el nombre de la capa esta limpio, simplemente se transfiere al nuevo diccionario para que se conserve tal cual
            model_dict[k] = v

    # cargan los pesos dentro del modelo vacio (strict=True) para que la red recupere todo lo que aprendio en el entrenamiento
    model.load_state_dict(model_dict, strict=True)
    
    # envia la red neuronal a la memoria de la tarjeta grafica para que procese las imagenes de prueba 
    model = model.cuda()

    logger.info('Loading test loader...')
    # carga el dataset de entrenamiento (metadatos) para que extraer el mismo abecedario que el modelo uso para aprender
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)

    # carga el conjunto de imagenes de prueba (test.ln) pasandole el alfabeto del entrenamiento (ralph) 
    test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    
    # empaquetan las imagenes de prueba en el DataLoader desactivando el modo aleatorio (shuffle=False) para que la evaluacion pase linea por linea
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    # inicia el traductor conectando el abecedario extraido para que convierta el texto de prueba humano a numeros comparables
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    
    # declara la funcion de error CTC para que pueda calcular la perdida residual durante la prueba
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    # apaga  el modo de entrenamiento de la red con eval() para que no se apagan neuronas durante la prueba
    model.eval()
    
    # bloquea el calculo de gradientes (no_grad) para que la tarjeta grafica no pierda tiempo ni memoria intentando deducir como aprender, enfocandose en transcribir
    with torch.no_grad():
        # llama a la funcion de validacion de valid.py pasandole todo el modelo preparado para que haga el calculo y de los errores finales
        val_loss, val_cer, val_wer, preds, labels = valid.validation(model,
                                                                     criterion,
                                                                     test_loader,
                                                                     converter)

    # imprimen los resultados CER y WER
    logger.info(
        f'Test. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} ')


if __name__ == '__main__':
    args = option.get_args_parser()
    main()