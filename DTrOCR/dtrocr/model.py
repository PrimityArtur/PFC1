import torch
from torch import nn, Tensor
from typing import Optional, Tuple, Dict, Any

from config import DTrOCRConfig
from processor import DTrOCRProcessor
from data import DTrOCRLMHeadModelOutput, DTrOCRModelOutput, DTrOCRProcessorOutput

from transformers.models.vit.modeling_vit import ViTPatchEmbeddings
from transformers.generation.logits_process import LogitsProcessorList
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

# nucleo Transformer para que la imagen y el texto se fusionen desde el inicio y pasen a traves de multiples capas de atencion 
class DTrOCRModel(nn.Module):
    # para que, al inicializar el modelo, se creen todas sus capas, redes neuronales y memorias internas antes de procesar datos
    def __init__(self, config: DTrOCRConfig):
        super().__init__()
        
        # Se crea la capa 'patch_embeddings' usando ViT para que tome la imagen de entrada, la corte en cuadritos (parches) y convierta esos pixeles en vectores 
        self.patch_embeddings = ViTPatchEmbeddings(config)
        
        # para que cada palabra o letra del texto se traduzca a un vector matematico del mismo tamaño que los parches de imagen
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        ## para que la red sepa en que orden van las cosas (cual parche va primero, cual letra despues), ya que la red por defecto no tiene nocion del orden espacial o temporal
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # apilan multiples bloques GPT2 ('hidden_layers') en una lista para que la informacion pase por varias fases de razonamiento
        self.hidden_layers = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
        # Se crea la capa dropout para que apague aleatoriamente algunas conexiones y obligue a la red a generalizar en lugar de memorizar exactamente los datos de entrenamiento
        self.dropout = nn.Dropout(config.attn_pdrop)
        
        # normalizacion para que estabilice la escala de los numeros al final de las capas, evitando que las sumas matematicas se salgan de control
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self._attn_implementation = config._attn_implementation

        # Se llama a la funcion de inicializacion de pesos para que el modelo inyecte el conocimiento preentrenado descargado de internet y no empiece el entrenamiento desde la ignorancia 
        self.initialise_weights(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
    ) -> DTrOCRModelOutput:
        
        # Se obtiene el dispositivo (CPU o GPU) de los datos de entrada para que todos los tensores generados internamente se envien al mismo chip y no ocurran errores de comunicacion
        device = input_ids.device if input_ids is not None else input_ids.device
        input_ids = input_ids.view(-1, input_ids.shape[-1])

        # Se verifica si hay 'past_key_values' (una memoria de calculos de pasos anteriores). Sirve para que, al momento de adivinar letra por letra, el modelo no tenga que recalcular toda la imagen de nuevo, ahorrando muchisimo tiempo de procesamiento
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.hidden_layers))
        else:
            past_length = past_key_values[0][0].size(-2)

        # Se extraen los embeddings de los parches de imagen y de texto. La condicion 'if past_length == 0' sirve para que SOLO procese la imagen en el primer paso. En los siguientes pasos, como la imagen ya esta en memoria, la omite
        patch_embeddings = self.patch_embeddings(pixel_values) if past_length == 0 else None
        token_embeddings = self.token_embedding(input_ids)

        # concatenan ambos embeddings (imagen + texto). Sirve para que se forme un unico grupo de datos continuo. El modelo vera primero los parches visuales y justo despues las letras, en la misma secuencia
        if patch_embeddings is not None:
            patch_and_token_embeddings = torch.concat([patch_embeddings, token_embeddings], dim=-2)
        else:
            patch_and_token_embeddings = token_embeddings
        input_shape = patch_and_token_embeddings.shape

        # generan los 'position_ids' para que se le pegue un numero de asiento (0, 1, 2, 3...) a cada parte del tren, indicandole al modelo donde esta parado exactamente cada parche y cada letra
        if position_ids is None or past_length == 0:
            position_ids = torch.arange(past_length, input_shape[1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = torch.ones_like(position_ids, device=position_ids.device) * past_length
        position_embeddings = self.positional_embedding(position_ids)

        # fusiona el contenido abstracto con su posicion espacial/temporal mediante una suma. para que la red entienda simultaneamente "que es esto" y "donde esta ubicado"
        hidden_states = patch_and_token_embeddings + position_embeddings
        
        # Se aplica la capa de descarte (dropout) para que se añada ruido y la red se haga mas robusta
        hidden_states = self.dropout(hidden_states)

        # manipula la mascara de atencion. para que se construya una mascara hibrida: le pega bloques de "unos" (1) a la parte de la imagen para que el modelo NUNCA se censure al ver la imagen, pero conserva las reglas estrictas del texto para que no haga trampa mirando palabras futuras
        if attention_mask is not None:
            attention_mask = torch.concat(
                [
                    torch.ones(
                        attention_mask.shape[0],
                        patch_embeddings.shape[-2] if patch_embeddings is not None else past_length,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    ),
                    attention_mask
                ], dim=-1
            )
            # Prepara la mascara segun el tipo de acelerador grafico para que el formato de los unos y ceros sea compatible con la formula de atencion de Hugging Face
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask=attention_mask,
                    input_shape=(input_shape[0], input_shape[-2]),
                    inputs_embeds=patch_and_token_embeddings,
                    past_key_values_length=past_length,
                )

        # recorre las capas ocultas (los 12 bloques GPT2). Cada bloque de atencion analiza las relaciones entre imagen y texto y hace la representacion mas profunda e inteligente
        presents = () if use_cache else None
        for hidden_layer, layer_past in zip(self.hidden_layers, past_key_values):
            outputs = hidden_layer(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            hidden_states = outputs[0] # Actualiza el tren de datos con la salida mas inteligente
            
            # Si se solicito usar cache, guarda los calculos de este bloque Sirve para que en la siguiente vuelta autoregresiva se ahorre calculo de este bloque
            if use_cache is True:
                presents = presents + (outputs[1],)

        # Se normalizan los datos despues de haber pasado por todos los bloques para que se suavicen los valores antes de mandarlos a la capa final de calificacion
        hidden_states = self.layer_norm(hidden_states)

        # Se empacan el estado oculto resultante y la memoria cache en la estructura de salida. Sirve para que el siguiente componente (DTrOCRLMHeadModel) lo reciba organizado y pueda calificar el error.
        return DTrOCRModelOutput(hidden_states=hidden_states, past_key_values=presents)

    # Sirve para que el modelo descargue los pesos ya entrenados por OpenAI y reemplace las conexiones neuronales con conocimiento real del idioma inlges, acelerando el entrenamiento final
    def initialise_weights(self, config: DTrOCRConfig) -> None:
        
        # Descarga el modelo GPT-2 base de la nube para una plantilla con pesos listos
        pretrained_gpt2 = GPT2Model.from_pretrained(config.gpt2_hf_model)

        # Bucle que copia los pesos capa por capa. para que transfiramos pesos de cada bloque del GPT-2 original directamente a  bloques GPT2Block recien creados
        for hidden_layer, pretrained_hidden_layer in zip(self.hidden_layers, pretrained_gpt2.h):
            hidden_layer.load_state_dict(pretrained_hidden_layer.state_dict())

        # Copia los pesos del diccionario de palabras (token embeddings) para que la red entienda el ingles de la misma forma que lo entendia el modelo original
        self.token_embedding.load_state_dict(pretrained_gpt2.wte.state_dict())

# calcula las probabilidades de las letras y medir el error durante el entrenamiento
class DTrOCRLMHeadModel(nn.Module):
    
    def __init__(self, config: DTrOCRConfig):
        super().__init__()
        self.config = config

        # instancia el modelo base para que toda la fusion de imagen y texto ocurra aqui adentro y devuelva el pensamiento abstracto
        self.transformer = DTrOCRModel(config)
        
        # Sirve para que tome el vector abstracto que sale del transformer (768 numeros) y lo transforme (lo expanda) a la cantidad exacta de palabras del diccionario (50257). Cada numero resultante sera la probabilidad de que esa letra sea la correcta.
        self.language_model_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # extraen los tamaños de la imagen y los parches
        image_size, patch_size = config.image_size, config.patch_size
        
        # Se calcula multiplicando ancho por alto de los parches. para que el modelo sepa exactamente cuantos "tokens" pertenecen a la imagen. Esto es vital para que, durante el calculo del error, el sistema ignore la imagen y solo califique a la red por las letras que adivino
        self.image_embedding_length = int((image_size[0] / patch_size[0]) * (image_size[1] / patch_size[1]))

    # Genera las predicciones (logits) y, si le pasas las respuestas correctas (labels), actua como maestro calificando el error (loss) y la precision (accuracy)
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        labels: Optional[torch.LongTensor] = None,
    ) -> DTrOCRLMHeadModelOutput:
        
        # Pasa los datos de entrada a traves del nucleo transformer para que se extraigan y procesen las caracteristicas combinadas de la imagen y el texto
        transformer_output = self.transformer(
            pixel_values=pixel_values,
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        
        # Pasa el pensamiento abstracto por la cabeza lineal. para que se generen los 'logits', que son las puntuaciones de cada posible letra del vocabulario
        logits = self.language_model_head(transformer_output.hidden_states)

        loss, accuracy = None, None
        
        # Inicia el bloque condicional si se proporcionaron 'labels' (el texto real) para que el codigo sepa que estamos en fase de entrenamiento y debe calcular que tan mal lo hizo la red
        if labels is not None:
            # Envia las etiquetas a la misma memoria grafica GPU que las predicciones
            labels = labels.to(logits.device)

            # Se recorta la matriz 'logits' quitando la parte inicial correspondiente a la imagen, para que la red no sea evaluada por lo que predijo mientras miraba los parches visuales, ya que ahi no hay texto que adivinar
            shift_logits = logits[..., self.image_embedding_length:-1, :].contiguous()
            
            # recorta la matriz 'labels' desplazandola una posicion hacia el futuro. para que la comparacion sea justa: compara "lo que el modelo creyo que seguia despues de la letra 1" contra "la letra 2 real"
            shift_labels = labels[..., 1:].contiguous()

            # Entropia Cruzada para que calcule un numero (error) que penaliza gigantescamente al modelo si estaba muy seguro de una letra incorrecta
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            
            # Se aplica la funcion a las predicciones y las etiquetas aplanando las matrices (view(-1)). para que calcule el error individual de cada token por separado
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Se crea la matriz comparando el indice mayor del logit (argmax) contra la etiqueta. para que devuelva puros Verdaderos (1) o Falsos (0) dependiendo de si la red atino a la letra exacta o no
            label_matches = shift_labels.view(-1) == torch.argmax(
                torch.nn.functional.softmax(shift_logits.view(-1, shift_logits.size(-1)), dim=-1), dim=-1
            )

            # Se aplica la mascara de atencion al calculo del error. para que los tokens de relleno vacio (pad_token) se multipliquen por cero, evitando que la red sea penalizada o premiada por adivinar espacios vacios al final de oraciones cortas
            if attention_mask is not None:
                mask = attention_mask[..., 1:].reshape(-1)

                loss = (mask * loss).sum() / mask.sum()
                accuracy = (mask * label_matches).sum() / mask.sum()
            else:
                loss = loss.mean()
                accuracy = torch.sum(label_matches) / label_matches.shape[0]

        # Retorna el paquete final de datos empaquetado. para que el bucle de entrenamiento extraiga el 'loss' y le diga a PyTorch que actualice los pesos (backward).
        return DTrOCRLMHeadModelOutput(
            loss=loss,
            logits=logits,
            accuracy=accuracy,
            past_key_values=transformer_output.past_key_values
        )

    # Generacion de texto. para que, cuando el modelo ya esta entrenado y solo recibe una imagen, organice las configuraciones y lance el bucle para adivinar toda la palabra
    @torch.no_grad() # Apaga el calculo de derivadas matemáticas. Sirve para ahorrar muchisima memoria RAM porque en inferencia no vamos a aprender nada nuevo
    def generate(
            self,
            inputs: DTrOCRProcessorOutput,
            processor: DTrOCRProcessor,
            num_beams: int = 1,
            use_cache: bool = True
    ):
        batch_size = inputs.input_ids.shape[0]
        model_kwargs = {
            'pixel_values': inputs.pixel_values,
            'attention_mask': inputs.attention_mask,
            'use_cache': use_cache
        }
        
        # agrupa los identificadores clave del diccionario. para que el bucle de generacion interno sepa cuando detenerse (eos_token_id), como rellenar huecos (pad_token_id) y cuantas iteraciones maximas puede dar
        generation_config = GenerationConfig(
            max_new_tokens=1,
            pad_token_id=processor.tokeniser.pad_token_id,
            eos_token_id=processor.tokeniser.eos_token_id,
            bos_token_id=processor.tokeniser.bos_token_id,
            num_beams=num_beams,
            max_length=processor.tokeniser.model_max_length
        )

        # Copia o clona los inputs de entrada dependiendo del 'num_beams'. para que, si el modelo va a explorar 3 caminos alternativos a la vez, se triplique la imagen en memoria para soportar esas ramas paralelas
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=inputs.input_ids,
            expand_size=generation_config.num_beams,
            **model_kwargs,
        )

        # Ejecuta la funcion que define las reglas de paro. para que el modelo no se quede en un bucle infinito atrapado adivinando letras 
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            processor=processor
        )

        # Decide que ruta algoritmica tomar basandose en el numero de rayos (beams)
        if num_beams > 1:
            # para que administre multiples oraciones a la vez, guardando en memoria solo las ramas de palabras con mejores probabilidades globales
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs.input_ids.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=LogitsProcessorList(),
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                **model_kwargs,
            )

        elif num_beams == 1:
            # Ejecuta la generacion simple (Greedy). para que avance lo mas rapido posible eligiendo ciegamente la letra con mayor probabilidad en ese preciso instante
            result = self._sample(
                input_ids,
                logits_processor=LogitsProcessorList(),
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                **model_kwargs,
            )
        else:
            raise ValueError("num_beams must be a positive integer.")

        return result

    #Bucle Autoregresivo (inferencia) para que la red recicle su propia salida. Adina una letra, la pega al final de su texto de entrada, y vuelve a analizar todo para adivinar la siguiente.
    def _sample(
        self,
        input_ids: torch.Tensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> torch.Tensor:
        
        pad_token_id = generation_config.pad_token_id
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        batch_size = input_ids.shape[0]
        
        # Se crea la matriz llena de unos (1). para que el bucle rastree que imagenes del lote aun siguen adivinando texto. Si terminan, se volvera cero.
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        this_peer_finished = False
        
        # Inicia el bucle mientras no hayan terminado. para que la red siga prediciendo letras 
        while not this_peer_finished:
            
            # Formatea los datos actuales para mandarlos a la red
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # Ejecuta el forward, obteniendo los outputs
            outputs = self(**model_inputs)

            # Extrae unicamente la prediccion de la ULTIMA letra generada (`[:, -1, :]`). para que el sistema ignore las letras pasadas (ya adivinadas) y solo enfoque su esfuerzo matematico en la letra nueva. .clone() se usa para aislar ese pedaso de memoria
            next_token_logits = outputs.logits[:, -1, :].clone()

            # Pre-procesa los puntajes segun reglas adicionales
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Selecciona la letra definitiva (`argmax`) para que busque en el diccionario de 50257 opciones cual fue la letra especifica que tuvo el puntaje ganador
            next_tokens = torch.argmax(next_token_scores, dim=-1)

            # Filtra las letras adivinadas por oraciones que ya acabaron. para que, si una palabra ya encontro su fin (EOS) en iteraciones pasadas, obligue a la red a seguir escupiendo tokens de relleno (pad) para no arruinar la oracion
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Concatena la letra nueva con todo el texto historico acumulado (`input_ids`). para que en la proxima vuelta del bucle 'while', la red lea la historia completa mas la letra nueva
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # Actualiza variables temporales y de posicion (`model_kwargs`). para que la memoria 'past_key_values' se pase a la siguiente iteracion, evitando reprocesar las imagenes
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

            # Modifica la bandera verificando si se encontro un simbolo final. para que si el nuevo token fue EOS, esta oracion pase de valer 1 a 0.
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            
            # Comprueba la condicion de salida del bucle infinito. para que, si todas las oraciones del lote actual llegaron a cero, se corte el bucle.
            this_peer_finished = unfinished_sequences.max() == 0

            # Elimina de la RAM el peso de los calculos crudos (`del outputs`). para que las matrices temporales masivas no saturen la tarjeta grafica (VRAM) despues de muchas vueltas
            del outputs

        # Devuelve la matriz completa con todo el texto generado desde principio a fin.
        return input_ids

    # Metodo de Busqueda en Haz para que la red no sea impulsiva. En lugar de elegir la mejor letra en cada paso (lo cual puede llevar a callejones sin salida), explora multiples caminos (rayos/beams) a la vez, sumando las probabilidades de la oracion completa, y al final escoge la frase que tenga mas sentido global
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> torch.Tensor:
        
        # Se extraen los identificadores de relleno (pad) y fin (eos) para que el algoritmo sepa con que tapar huecos y cuando un camino ya termino
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id

        # extrae el tamaño del lote y la cantidad de rayos a explorar
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        # extrae el tamaño total expandido y la longitud actual de la oracion
        batch_beam_size, cur_len = input_ids.shape
        
        # inicializa el rastreador de posiciones para la memoria cache
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # Se verifica que las dimensiones matematicas cuadren para que el programa lance un error claro si olvidaste multiplicar los inputs por el numero de rayos antes de llamar a esta funcion
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # Se crea una matriz de puntajes iniciales con ceros para el primer rayo y -infinito (-1e9) para el resto. para que, en el primerisimo paso, todos los rayos no intenten adivinar exactamente la misma primera letra. Fuerza a que cada rayo explore una letra de inicio diferente
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False
        decoder_prompt_len = input_ids.shape[-1]
        
        # exploracion paralela para que la red siga extendiendo las ramas del arbol de palabras hasta que todos los caminos terminen
        while not this_peer_finished:
            
            # Formatea los datos de entrada para la siguiente iteracion
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # Ejecuta la red neuronal
            outputs = self(**model_inputs)

            # Extrae las predicciones de la ultima letra generada y las clona. para aislar la memoria y poder descartar el resto del bloque masivo de 'outputs' luego
            next_token_logits = outputs.logits[:, -1, :].clone()
            
            # Convierte las predicciones a probabilidades logaritmicas (log_softmax). para que, al evaluar una oracion larga, podamos SUMAR las probabilidades de las letras en lugar de MULTIPLICARLAS (lo que causaria que el numero se vuelva cero por limites matematicos de la computadora)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            # Aplica procesadores logicos a las puntuaciones (penalizar si repite mucho una letra)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            
            # Suma el puntaje acumulado de la oracion hasta ahora con el puntaje de la nueva letra. para que el sistema evalue el "peso" total de ese camino o rama.
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Reestructura la matriz de puntuaciones para aplanar los rayos. para que PyTorch pueda buscar los mejores puntajes absolutos entre TODOS los caminos cruzados.
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # calcula cuantos tokens rescatar (usualmente 2 veces el numero de rayos).
            n_tokens_to_keep = max(2, 1 + 1) * num_beams
            
            # usa la funcion 'topk' para sacar las opciones con puntaje mas alto. para que se poden las ramas debiles del arbol y solo sobrevivan los mejores prospectos de oracion.
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
            )

            # aplican trucos matematicos (division y modulo) sobre los indices seleccionados para que el sistema averigue dos cosas de un solo numero: de que RAYO (rama) provino esta buena letra (next_indices), y que LETRA exacta del diccionario es (next_tokens).
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # pasa toda la informacion al administrador de rayos. para que la herramienta evalue si algun rayo llego al fin (eos) y lo guarde en la boveda de oraciones terminadas
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                decoder_prompt_len=decoder_prompt_len,
            )

            # Se extraen los estados actualizados del administrador
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            # Se une la nueva letra ganadora a la historia de su rayo correspondiente
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # Se actualiza el cache y posiciones para la siguiente iteracion
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

            # Se borra la variable outputs pesada de la memoria RAM para evitar el colapso de la tarjeta grafica (Out of Memory)
            del outputs

            # Si estamos usando memoria cache, reorganiza el historial usando el indice ganador. para que, si un rayo murio y otro rayo se dividio en dos, la memoria cache se copie y se asigne a las nuevas ramas correctamente, para que no mezclen sus recuerdos
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            cur_len = cur_len + 1

            # Comprueba si todos los rayos ya terminaron o chocaron con el limite
            if beam_scorer.is_done or all(stopping_criteria(input_ids, None)):
                this_peer_finished = True

        # Al terminar el bucle, evalue lo almacenado y devuelva la mejor oracion absoluta
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            decoder_prompt_len=decoder_prompt_len,
        )

        return sequence_outputs["sequences"]

    # para recopilar las reglas de detencion. para crear una lista de condiciones que, si se cumplen, y al bucle de generacion que se detenga inmediatamente
    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        processor: Optional[DTrOCRProcessor] = None,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        
        # Regla 1 Limite de longitud. para que no escriba mas palabras del maximo permitido
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        # Regla 2 Limite de tiempo. Sirve para que aborte si se tarda demasiados segundos
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
            
        # Regla 3 Palabras prohibidas. Sirve para que corte si genera una palabra especifica
        if generation_config.stop_strings is not None:
            if processor is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            criteria.append(StopStringCriteria(
                stop_strings=generation_config.stop_strings, tokenizer=processor.tokeniser)
            )
            
        # Regla 4 Token de fin de oracion (EOS). Sirve para que pare cuando decida que la oracion termino de forma natural.
        if generation_config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
            
        return criteria

    # reorganizar la memoria Cache para que el Beam Search manipule el cerebro temporal de la red, copiando y moviendo la memoria de las ramas vivas y sobreescribiendo la memoria de las muertas.
    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> tuple[tuple[Tensor, ...], ...]:
        # Itera sobre todas las capas y reordena (index_select) la memoria segun los rayos que ganaron (beam_idx).
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    # actualiza los argumentos antes del siguiente ciclo. para preparar las variables dinamicas (memoria, mascaras) para el siguiente bucle del while, añadiendoles un espacio extra para la nueva letra
    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: DTrOCRLMHeadModelOutput,
        model_kwargs: Dict[str, Any],
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:

        # Guarda la memoria fresca escupida por el modelo
        model_kwargs['past_key_values'] = outputs.past_key_values

        # Le suma un "1" a la mascara de atencion para que la red sepa que el texto ahora es una letra mas largo y debe prestarle atencion a esa nueva posicion
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # Actualiza el indice temporal para decirle al sistema en que posicion exacta del tren de datos debe insertar la siguiente letra en la proxima vuelta
        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        return model_kwargs

    # recortar la entrada antes de mandarla a la red para optimizar la velocidad. Si la red ya recuerda todo el texto pasado (cache), le recorta la historia y le manda unicamente la ultima letra para evaluarla
    @staticmethod
    def prepare_inputs_for_generation(
        input_ids: torch.Tensor, past_key_values=None, **kwargs
    ) -> Dict[str, Any]:
        
        # Si existe memoria (past_key_values), significa que no estamos en el primer paso
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Recorta el input_ids dejando solo los elementos nuevos que la memoria aun no ha procesado
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            # Reemplaza la gran oracion por un vector pequeño
            input_ids = input_ids[:, remove_prefix_length:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # Si no nos pasaron position_ids, los creamos. para que la red sepa la ubicacion matematica real de esa unica letra nueva que le estamos pasando, respecto a toda la oracion que esta en memoria
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        # Empaqueta y devuelve los datos listos para el 'forward'.
        model_inputs = {
            'input_ids': input_ids,
            "past_key_values": past_key_values,
            'pixel_values': kwargs['pixel_values'],
            'use_cache': kwargs.get("use_cache"),
            'labels': kwargs.get("labels"),
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

        return model_inputs

    # inicializar la posicion del Cache. para decirle al sistema en que indice debe empezar a guardar la memoria cuando arranque la generacion por primera vez.
    @staticmethod
    def _get_initial_cache_position(input_ids, model_kwargs):
        if not model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = None
            return model_kwargs

        # Genera un arreglo de numeros secuenciales del tamaño del input inicial
        model_kwargs["cache_position"] = torch.arange(0, input_ids.shape[-1], device=input_ids.device)
        return model_kwargs

    # duplicar tensores. para que, si el Beam Search requiere explorar 3 caminos paralelos, este metodo agarre la imagen y el input inicial y lo triplique en la memoria de la tarjeta grafica para que se procesen a la vez 
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: Optional[torch.LongTensor],
        expand_size: int = 1,
        **model_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        # Funcion interna para multiplicar todos los tensores dentro de un diccionario
        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                        key != "cache_position"
                        and dict_to_expand[key] is not None
                        and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    # Multiplica el tensor repitiendolo (repeat_interleave)
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # Multiplica la oracion inicial
        input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        
        # Multiplica el resto de los argumentos (como la imagen y la mascara)
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        return input_ids, model_kwargs