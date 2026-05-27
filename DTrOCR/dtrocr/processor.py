from transformers import GPT2Tokenizer, AutoImageProcessor

from PIL import Image
from typing import List, Union

from config import DTrOCRConfig
from data import DTrOCRProcessorOutput


# para que actue como interprete del modelo, tomando las imagenes PNG y el texto en digitalizado, y convirtiendo ambos en tensores que la red neuronal pueda procesar 
class DTrOCRProcessor:
    
    # para que al inicializar este procesador, ya tenga configurados en memoria tanto el traductor visual como el traductor de texto con las reglas y tamaños correctos
    def __init__(self, config: DTrOCRConfig, add_bos_token: bool = False, add_eos_token: bool = False):
        
        # Se inicializa el procesador de imagenes ViT usando la configuracion dada, para que sepa exactamente a que tamaño (altura y anchura) debe estirar o encoger las imagenes reales antes de extraer sus pixeles
        self.vit_processor = AutoImageProcessor.from_pretrained(
            config.vit_hf_model,
            size={
                "height": config.image_size[0],
                'width': config.image_size[1]
            },
            use_fast=True # Activa la version rapida para que consuma menos tiempo procesando las fotos
        )
        
        # Se inicializa el tokenizador de texto GPT2 para que sepa como usar su diccionario interno y dividir palabras en numeros (tokens)
        self.tokeniser = GPT2Tokenizer.from_pretrained(
            config.gpt2_hf_model,
            add_bos_token=add_bos_token,
            # Se calcula el espacio limite para el texto, restando la cantidad de parches de imagen a la capacidad total de la red, para que el tokenizador deje exactamente el espacio vacio que sobra solo para las palabras, evitando asi que la red colapse por exceso de datos
            model_max_length=config.max_position_embeddings - int(
                (config.image_size[0] / config.patch_size[0]) * (config.image_size[1] / config.patch_size[1])
            )
        )
        
        # iguala el token de relleno (pad) al token de inicio (bos), para que el modelo sepa con que simbolo rellenar los espacios vacios cuando una palabra es muy corta, sin inventar un simbolo nuevo que confunda a la red
        self.tokeniser.pad_token = self.tokeniser.bos_token
        
        # guarda la instruccion de agregar o no el token de fin de secuencia, para que el modelo sepa cuando debe dejar de intentar adivinar mas letras
        self.tokeniser.add_eos_token = add_eos_token

        # Se reemplaza una funcion interna del tokenizador de Hugging Face por la definida, para que en lugar de usar su comportamiento estandar, se vea obligado a usar la funcion personalizada (definida mas abajo) para agregar tokens especiales
        self.tokeniser.build_inputs_with_special_tokens = modified_build_inputs_with_special_tokens.__get__(
            self.tokeniser
        )

    # procesamiento, para que cuando le pasemos imagenes y textos juntos, el procesador los separe, convierta cada uno por su lado, y luego empaquete todos los resultados matematicos para enviarlos al modelo
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = False,
        *args,
        **kwargs
    ) -> DTrOCRProcessorOutput:
        
        # procesa el texto ingresado utilizando el tokenizador, para que convierta las palabras legibles en matrices de numeros enteros (input_ids) y genere sus mascaras de atencion, haciendolo solo si el usuario envio un texto
        text_inputs = self.tokeniser(
            texts, padding=padding, *args, **kwargs
        ) if texts is not None else None

        # procesan las imagenes ingresadas utilizando el procesador visual, para que normalice los colores y convierta los pixeles en tensores (pixel_values), haciendolo solo si el usuario envio una imagen
        image_inputs = self.vit_processor(
            images, input_data_format=input_data_format, *args, **kwargs
        ) if images is not None else None

        # ensambla y retorna la caja de datos empaquetada (DTrOCRProcessorOutput), para que el modelo reciba los datos donde los espera
        return DTrOCRProcessorOutput(
            # Guarda los pixeles listos para que la red los vea
            pixel_values=image_inputs["pixel_values"] if images is not None else None,
            
            # Guarda los numeros del texto para que la red los lea
            input_ids=text_inputs['input_ids'] if texts is not None else None,
            
            # Guarda los unos y ceros para que la red sepa a que partes prestar atencion
            attention_mask=text_inputs['attention_mask'] if texts is not None else None,
            
            # clonan los inputs del texto, para que, durante el entrenamiento, la funcion de error (Cross Entropy) tenga la respuesta correcta guardada y pueda calificar que tan bien lo hizo la red
            labels=text_inputs['input_ids'] if texts is not None and return_labels else None
        )


# Funcion que modifica como se arman los tokens especiales en el texto. Sirve para que se inserten manualmente etiquetas delimitadoras (inicio y fin) al texto original, para que la red sepa exactamente en que punto del tensor ya acabo de procesar la imagen  y va a empezar a adivinar la primera letra, y en que punto ya termino la oracion
def modified_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    
    # Se verifica si la configuracion general exigio agregar un token de inicio (BOS). Si es asi, se crea una lista con el numero clave que representa (aqui empieza el texto), para que se coloque justo antes de la primera letra
    if self.add_bos_token:
        bos_token_ids = [self.bos_token_id]
    else:
        bos_token_ids = []

    # verifica si la configuracion exigio agregar un token de fin (EOS). Si es asi, se crea una lista con el numero clave que representa (aqui termina el texto), para que se coloque justo despues de la ultima letra.
    if self.add_eos_token:
        eos_token_ids = [self.eos_token_id]
    else:
        eos_token_ids = []

    # concatenan las tres listas: [Inicio] + [Texto Real] + [Fin], para que quede un solo grupo de numeros continuo que la red leera paso a paso
    output = bos_token_ids + token_ids_0 + eos_token_ids

    # Si no hay una segunda secuencia de texto auxiliar (lo cual es lo normal en el proyecto), se detiene la funcion y devuelve el output ya ensamblado para que pase directo al modelo
    if token_ids_1 is None:
        return output

    # En caso de que se enviara un segundo bloque de texto, lo une al final de la cola, para que la libreria no falle en tareas que requieren comparar dos oraciones distintas
    return output + bos_token_ids + token_ids_1