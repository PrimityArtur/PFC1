import torch
import numpy as np

from dataclasses import dataclass
from typing import Optional, Union, List

# estructura de datos de la arquitectura Transformer, para que el programa tenga una caja de envio estandarizada donde guardar los resultados profundos del modelo antes de convertirlos en letras
@dataclass
class DTrOCRModelOutput:
    # para que almacene el resultado numerico despues de que la imagen y el texto se fusionan
    hidden_states: torch.FloatTensor
    # guarda una memoria cache de lo que el modelo ya penso en pasos anteriores, para que en el siguiente paso de lectura no tenga que recalcular todo desde cero
    past_key_values: torch.FloatTensor


# Define la estructura de datos final despues de pasar por la capa de clasificacion, para que empaquete las predicciones finales y las metricas de error que serviran tanto para entrenar la red como para leer el texto 
@dataclass
class DTrOCRLMHeadModelOutput:
    
    # matriz logits para que contenga las puntuaciones de cada letra posible, para que el sistema sepa que letra tiene mayor probabilidad de ser la correcta
    logits: torch.FloatTensor
    
    # variable opcional loss para que calcule y guarde el error total, para que el optimizador sepa cuanto se equivoco la red y pueda corregir sus pesos 
    loss: Optional[torch.FloatTensor] = None
    
    # accuracy para que registre el porcentaje de aciertos, para que se pueda ver si el modelo esta aprendiendo durante el entrenamiento
    accuracy: Optional[torch.FloatTensor] = None
    
    # para que la funcion principal pueda exportar esta memoria temporal y la use en el bucle de generacion de nuevas palabras
    past_key_values: Optional[torch.FloatTensor] = None


# estructura de los datos de entrada preparados, para que agrupe en un solo paquete la imagen y el texto digitalizado ya transformados en numeros, para que el modelo pueda procesarlos
@dataclass
class DTrOCRProcessorOutput:
    
    # matriz para que guarde los pixeles de la imagen real ya normalizados y recortados, para que la red ViT pueda empezar a buscar patrones de caligrafia
    pixel_values: Optional[torch.FloatTensor] = None
    
    # matriz para que guarde el texto real convertido en tokens (numeros de diccionario), para que el modelo de lenguaje GPT pueda leer el contexto de lo que ya esta escrito
    input_ids: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
    
    # matriz compuesta de unos y ceros para que guie la atencion del modelo, para que el modelo sepa exactamente que datos son importantes (1) y cuales son relleno vacio (0) que debe ignorar
    attention_mask: Optional[Union[torch.FloatTensor, np.ndarray, List[int]]] = None
    
    # matriz para que almacene las respuestas correctas (el texto original desplazado), para que la funcion de entropia cruzada actue como maestro y compare la prediccion del modelo con el texto digitalizado
    labels: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None