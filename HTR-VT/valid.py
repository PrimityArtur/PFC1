import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from utils import utils
import editdistance

# Congela el modelo, le inyecta un lote de imagenes de validacion/prueba, decodifica sus predicciones y compara el texto que la maquina creyo leer contra el texto escrito, devolviendo las tasas de error (CER y WER)
def validation(model, criterion, evaluation_loader, converter):

    # inicializan las variables de distancia de normalizada en cero para que los errores individuales por oracion puedan ser evaluados
    norm_ED = 0
    norm_ED_wer = 0

    # inicializan los acumuladores totales de distancia de edicion (tot_ED) en cero para que sumen el error absoluto de todo el dataset de prueba
    tot_ED = 0
    tot_ED_wer = 0

    # inicializan contadores de perdida (loss) y longitud de caracteres/palabras reales para que al final se pueda calcular un promedio 
    valid_loss = 0.0
    length_of_gt = 0
    length_of_gt_wer = 0
    count = 0
    all_preds_str = []
    all_labels = []

    # bucle sobre el evaluation_loader para que el modelo procese el conjunto de prueba en lotes 
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        # extrae la cantidad de imagenes en el lote actual para que las matrices de decodificacion sepan cuantas oraciones estan procesando simultaneamente
        batch_size = image_tensors.size(0)
        # envian los tensores de las imagenes a la tarjeta grafica (cuda) para que las predicciones se calculen a la maxima velocidad del hardware
        image = image_tensors.cuda()

        # llama al convertidor para que traduzca las etiquetas de texto reales a numeros enteros, permitiendo que la funcion de perdida las compare matematicamente
        text_for_loss, length_for_loss = converter.encode(labels)

        # inyectan las imagenes en el modelo para que la red neuronal genere sus mapas de probabilidades de caracteres (predicciones)
        preds = model(image)
        # fuerza a que las matrices de predicciones sean float para que no se pierda precision durante el calculo de errores
        preds = preds.float()
        # crea un tensor con la longitud de las secuencias visuales predichas para que el algoritmo CTC sepa cual es el limite util de cada vector antes del relleno
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        # reordenan las dimensiones (permute) y se aplica log_softmax para que las puntuaciones se conviertan en probabilidades logaritmicas normalizadas, formato para CTCLoss
        preds = preds.permute(1, 0, 2).log_softmax(2)

        # desactiva la aceleracion cuDNN temporalmente para que PyTorch no arroje errores de incompatibilidad interna al calcular la perdida CTC
        torch.backends.cudnn.enabled = False
        # calcula el error (cost) comparando las probabilidades predichas contra los textos reales, y se promedia (mean) para que represente la falla global del lote actual
        cost = criterion(preds, text_for_loss, preds_size, length_for_loss).mean()
        # vuelve a reactivar cuDNN para que el resto del sistema siga ejecutandose 
        torch.backends.cudnn.enabled = True

        # extrae el indice con el valor mas alto (max) en el eje del vocabulario para que el modelo elija la letra con mayor probabilidad como su decision final
        _, preds_index = preds.max(2)
        # transpone la matriz resultante y se aplana (view(-1)) para que quede una sola secuencia continua unidimensional de numeros ganadores
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        # pasa esta secuencia de numeros al decodificador para que los traduzca de vuelta a caracteres legibles, colapsando repetidos y borrando espacios en blanco CTC
        preds_str = converter.decode(preds_index.data, preds_size.data)
        
        # suma el error numerico escalar (item) al acumulador general para que guarde el historial de perdida de este lote
        valid_loss += cost.item()
        # incrementa el contador de lotes para que sirva de divisor en el promedio final
        count += 1

        # guardan las cadenas de texto predichas y las reales en las listas maestras para que se tenga un reporte completo de todo el conjunto de prueba
        all_preds_str.extend(preds_str)
        all_labels.extend(labels)

        # inicia un bucle emparejando (zip) cada texto predicho con su contraparte original para que se evalue el nivel de error letra por letra
        for pred_cer, gt_cer in zip(preds_str, labels):
            # usa la funcion eval de editdistance para que cuente cuantas inserciones, borrados o sustituciones son necesarias para convertir la prediccion en el texto 
            tmp_ED = editdistance.eval(pred_cer, gt_cer)
            # condiciona por si el texto original estaba vacio para que no ocurra un error matematico de division por cero
            if len(gt_cer) == 0:
                norm_ED += 1
            else:
                # normaliza la distancia dividiendola por la longitud real para que un error en una palabra muy larga no penalice igual que un error en una palabra muy corta
                norm_ED += tmp_ED / float(len(gt_cer))
            
            # acumula la cantidad de fallos de caracteres (tot_ED) y la cantidad de caracteres totales reales (length_of_gt) para que calculen el CER global
            tot_ED += tmp_ED
            length_of_gt += len(gt_cer)

        # inicia otro bucle de emparejamiento para que esta vez evalue los errores agrupando palabras enteras en lugar de caracteres
        for pred_wer, gt_wer in zip(preds_str, labels):
            # aplica la funcion de formateo a la prediccion para que aísle los signos de puntuacion con espacios y no manche la evaluacion de las palabras adyacentes
            pred_wer = utils.format_string_for_wer(pred_wer)
            # aplica el mismo formateo al texto original (Ground Truth) para que ambas cadenas sean justas y comparables
            gt_wer = utils.format_string_for_wer(gt_wer)
            
            # dividen las cadenas usando espacios (split) para que se conviertan en arreglos (listas) compuestos puramente de palabras
            pred_wer = pred_wer.split(" ")
            gt_wer = gt_wer.split(" ")
            
            # calcula la distancia de edicion entre las listas de palabras para que detecte si el modelo omitio, altero o invento palabras enteras
            tmp_ED_wer = editdistance.eval(pred_wer, gt_wer)

            # protege contra divisiones por cero en caso de listas de palabras vacias
            if len(gt_wer) == 0:
                norm_ED_wer += 1
            else:
                # normaliza el error de palabras dividiendolo entre la cantidad de palabras de la oracion real
                norm_ED_wer += tmp_ED_wer / float(len(gt_wer))

            # acumula el total de palabras erroneas y el conteo de palabras reales para que generen el WER global 
            tot_ED_wer += tmp_ED_wer
            length_of_gt_wer += len(gt_wer)

    # calcula la perdida final dividiendo el error acumulado entre el numero de lotes procesados para que devuelva el promedio 
    val_loss = valid_loss / count
    
    # divide el total de errores de edicion de caracteres entre el total de caracteres reales para que se obtenga el Character Error Rate (CER) final
    CER = tot_ED / float(length_of_gt)
    
    # divide el total de errores a nivel de palabra entre el total de palabras reales para que se calcule el Word Error Rate (WER)
    WER = tot_ED_wer / float(length_of_gt_wer)

    # retornan las tres metricas  mas las listas de textos para que el archivo train.py decida si este es el mejor modelo y debe guardarlo
    return val_loss, CER, WER, preds_str, labels