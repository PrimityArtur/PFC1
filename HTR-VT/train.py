import os
import json
import valid

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from utils import sam
from utils import option

from data import dataset
from model import HTR_VT

from functools import partial

# Toma la imagen, la pasa por la red activando las mascaras, y calcula el error CTC
def compute_loss(args, model, image, batch_size, criterion, text, length):
    # pasa la imagen al modelo activando la mascara de tramos (use_masking=True) para que el Transformer se vea obligado a adivinar partes borradas de la caligrafia
    preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    
    # convierte el tensor de salida a formato de punto flotante para que no haya errores de precision al calcular decimales muy pequeños
    preds = preds.float()
    
    # crea un vector que contiene la longitud de los tensores de prediccion multiplicandolo por el batch_size para que el algoritmo CTC sepa donde terminan los datos reales
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
    
    # reordenan las dimensiones y se aplica log_softmax para que las puntuaciones crudas se conviertan en probabilidades normalizadas listas para la funcion CTC
    preds = preds.permute(1, 0, 2).log_softmax(2)

    # desactiva la aceleracion cuDNN para que PyTorch no falle al procesar tensores de longitud variable en la funcion CTC
    torch.backends.cudnn.enabled = False
    
    # ejecuta la funcion de perdida (criterion) comparando la prediccion con el texto real y se promedia (mean) para que el modelo sepa cuanto se equivoco en este lote exacto
    loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
    
    # vuelve a encender cuDNN para que el resto del codigo corra con GPU
    torch.backends.cudnn.enabled = True
    
    # devuelve el numero total de error para que el optimizador pueda usarlo
    return loss


# Ensambla todas las piezas y controla el bucle de las 100,000 iteraciones de aprendizaje
def main():
    # llama a option.py para que lea todos los parametros que enviaste por la consola (batch_size, iteraciones, rutas)
    args = option.get_args_parser()
    
    # fija la semilla de aleatoriedad global (123) para que, si vuelves a correr el codigo, las imagenes se desordenen y alteren igual y sea reproducible
    torch.manual_seed(args.seed)
    # crea la ruta final sumando el directorio de salida (output) y el nombre del experimento (iam) para que sepa donde guardar
    args.save_dir = os.path.join(args.out_dir, args.exp_name)    
    # crea la carpeta fisicamente en el disco duro usando makedirs para que no de error al intentar guardar los primeros archivos
    os.makedirs(args.save_dir, exist_ok=True)
    # instancia el sistema de registro logger para que empiece a registrar los procesos en el archivo run.log
    logger = utils.get_logger(args.save_dir)    
    # imprimen todos los argumentos de consola en formato JSON para que quede un registro de como se configuro el experimento
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))    
    # inicia TensorBoard (SummaryWriter) para que empiece a grabar los numeros y luego ver las graficas del entrenamiento 
    writer = SummaryWriter(args.save_dir)

    # llama a HTR_VT para que ensamble la ResNet18 y el Transformer con las dimensiones de imagen requeridas
    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    # cuentan y suman todos los parametros entrenables del modelo para que sepas cuanta memoria exigira (53.4 millones)
    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    # enciende el modo entrenamiento (model.train) para que se activen funciones como el Dropout (apagado aleatorio de neuronas) y el enmascaramiento
    model.train()
    
    # manda el modelo completo a la tarjeta de video (cuda) para que los calculos se aceleren usando los nucleos de la GPU
    model = model.cuda()
    
    # crea la copia (EMA) para que vaya promediando  los pesos de la red a medida que aprende, garantizando evaluaciones estables
    model_ema = utils.ModelEma(model, args.ema_decay)
    
    # limpian los gradientes de la red para que no tenga basura acumulada en la memoria antes de empezar
    model.zero_grad()

    logger.info('Loading train loader...')
    # lee la lista de archivos de entrenamiento (train.ln) para que el dataset sepa donde estan las fotos
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    
    # crea el DataLoader pasandole la funcion SameTrCollate para que agrupe las imagenes de 32 en 32, les aplique las deformaciones y las devuelva en bloque paralelo
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    
    # envuelve el DataLoader en un generador infinito (cycle_data) para que el for-loop no se rompa cuando se acaben las imagenes, sino que vuelva a empezar automaticamente
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    # carga la lista de validacion para que el modelo evalue con datos que no uso para entenar
    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)    
    # crea el DataLoader de validacion sin deformaciones (shuffle=False y sin collate_fn) para que la evaluacion sea estandar
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    # inicializa el optimizador SAM envolviendo al AdamW para que las actualizaciones busquen valles de error
    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)    
    # declara la funcion de error CTCLoss para que mida que tan mala es la alineacion de los caracteres
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    
    # inicia el traductor CTCLabelConverter para que convierta las letras "a,b,c" a numeros "1,2,3" y la funcion CTC pueda procesarlas
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    # establecen las mejores puntuaciones iniciales (best_cer y best_wer) en un millon para que cualquier resultado de la primera prueba pase el record y se guarde
    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0

    #### train y eval ####

    # Inicia el bucle iterando desde 1 hasta el limite  100,000 para que la red entrene 
    for nb_iter in range(1, args.total_iter+1):

        # llama a la funcion coseno para que suba o baje la velocidad de aprendizaje segun el numero de iteracion actual
        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        # limpian los gradientes del optimizador para que los calculos matematicos del paso anterior no contaminen este paso
        optimizer.zero_grad()
        
        # extrae un nuevo bloque de imagenes deformadas y sus textos (batch) llamando a 'next' para que la red procese
        batch = next(train_iter)
        
        # mandan las imagenes a la GPU para que se procesen rapido
        image = batch[0].cuda()
        
        # traduce el texto de este bloque a formato numerico para que se pueda comparar en la ecuacion de perdida
        text, length = converter.encode(batch[1])
        batch_size = image.size(0)
        
        # forward se calcula el error ejecutando la red para que el sistema sepa como esta rindiendo
        loss = compute_loss(args, model, image, batch_size, criterion, text, length)        
        # backward se viaja hacia atras por la red neuronal calculando la derivada de los pesos para que el optimizador sepa hacia donde esta el pico del error.
        loss.backward()        
        # SAM. El optimizador SAM empuja los pesos intencionalmente hacia el pico del error para que la red evalue el peor escenario local
        optimizer.first_step(zero_grad=True)
        
        # forward. Se vuelve a pasar la misma imagen y calcular el error estando en el pico de la alto para que la red mida la pendiente
        compute_loss(args, model, image, batch_size, criterion, text, length).backward()        
        # SAM regresa a la posicion original y usa la informacion del pico bajo para dar un paso seguro lejos del error, actualizando los pesos
        optimizer.second_step(zero_grad=True)        
        # Se vuelve a limpiar la red entera para que el proximo lote no herede las matematicas de este.
        model.zero_grad()
        
        # le avisa a la copia (EMA) para que asimile los resutlados que acaba de obtener de la red
        model_ema.update(model, num_updates=nb_iter / 2)
        
        # suma el error de esta iteracion al total acumulado para que promediarlo
        train_loss += loss.item()

        # si el numero de iteracion es divisible entre print_iter (ej. 100), se detiene para que imprima informacion a la pantalla
        if nb_iter % args.print_iter == 0:
            # promedia el error de los ultimos 100 pasos
            train_loss_avg = train_loss / args.print_iter

            # Imprime en la consola el numero de paso, la tasa de aprendizaje y el error promedio
            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

            # guarda estos mismos numeros en TensorBoard para que dibuje las lineas de progreso
            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            
            # Reinicia la sumatoria a cero para que el proximo promedio 
            train_loss = 0.0

        # Si la iteracion actual es divisible entre eval_iter (ej. 1000), el modelo se detiene para evaluar
        if nb_iter % args.eval_iter == 0:
            # Apaga el modo entrenamiento para que la evaluacion sea estable (sin apagar neuronas por dropout)
            model.eval()
            
            # bloquea los gradientes (no_grad) para que el modelo sepa que no esta aprendiendo durante la evaluacion
            with torch.no_grad():
                # Llama al archivo valid.py pasandole LA RED copia (model_ema.ema) para que califique la version mas estable del modelo
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema, criterion, val_loader, converter)

                # Si el Error de Caracteres (CER) obtenido es menor al mejor CER historico, significa que el modelo mejoro, para que proceda a guardarlo
                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    
                    # Empaqueta los pesos principales, los pesos sombra y el optimizador en un diccionario para que pueda pausar y reanudar 
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    # Guarda el archivo  'best_CER.pth' 
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                # Si el Error de Palabras (WER) rompe su propio record, hace  el mismo procedimiento para que guarde una version optimizada para palabras completas.
                if val_wer < best_wer:
                    logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                # guarda los resultados de la evaluacion en TensorBoard 
                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                
                # Enciende de nuevo el modo entrenamiento para que el modelo siga aprendiendo en la siguiente iteracion 
                model.train()

if __name__ == '__main__':
    main()