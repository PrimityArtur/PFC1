from typing import Optional, Union, Tuple, List, Literal

# La clase de configuraciones para DTrOCR
# Esta clase actua como el "cerebro" o diccionario de ajustes. Almacena todos los hiperparametros para que el modelo se construya con las dimensiones y reglas 
class DTrOCRConfig:
    def __init__(
        self,
        # define el modelo base de lenguaje para que el sistema sepa que pesos preentrenados de GPT-2 descargar.
        gpt2_hf_model: str = 'openai-community/gpt2',        
        # define el modelo base de vision para que el sistema sepa que pesos preentrenados usar para extraer caracteristicas de la imagen
        vit_hf_model: str = 'google/vit-base-patch16-224',        
        
        # establece el tamaño del vocabulario para que la capa final sepa cuantas palabras/caracteres posibles puede predecir
        vocab_size: Optional[int] = 50257,
        
        # numero maximo de posiciones 256 para que el modelo sepa la longitud maxima de la secuencia (imagen + texto) que puede procesar
        max_position_embeddings: Optional[int] = 256,
        
        # define el tamaño oculto 768 para determinar la dimension de los vectores matematicos que fluyen dentro de la red
        hidden_size: Optional[int] = 768,
        
        # cantidad de capas ocultas para la profundidad del modelo GPT-2 (cuantos bloques Transformer se apilaran)
        num_hidden_layers: Optional[int] = 12,
        
        # Se indica el numero de cabezales de atención para que el modelo pueda enfocarse en multiples relaciones texto-imagen simultáneamente
        num_attention_heads: Optional[int] = 12,
        
        # tamaño del parche visual (4x8) para dividir la imagen de texto (que es alargada) en tokens visuales
        patch_size: Optional[Union[Tuple[int], List[int]]] = (4, 8),      
        
        # resolucion de entrada de la imagen (32x128) para asegurar que todas las imagenes se redimensionen a la misma forma antes de entrar a la red
        image_size: Optional[Union[Tuple[int], List[int]]] = (32, 128),   
        
        # numero de canales (3 para RGB) para que el extractor de parches sepa como leer los colores de la imagen original
        num_channels: Optional[int] = 3,
        
        # probabilidades de "dropout" (0.1 o 10%) para apagar neuronas aleatoriamente durante el entrenamiento y así evitar el sobreajuste 
        resid_pdrop: Optional[float] = 0.1,
        embd_pdrop: Optional[float] = 0.1,
        attn_pdrop: Optional[float] = 0.1,
        
        # epsilon (1e-5) para usarlo en la Layer Normalization, evitando divisiones por cero en los calculos internas
        layer_norm_epsilon: Optional[float] = 1e-5,
        
        # elige la implementación de la atención (sdpa) para optimizar el uso de memoria y la velocidad en la GPU
        attn_implementation: Literal['sdpa', 'flash_attention_2'] = 'sdpa'
    ):
        # parametros recibidos como atributos propios de la clase para que model.py o processor.py puedan acceder 
        self.gpt2_hf_model = gpt2_hf_model
        self.vit_hf_model = vit_hf_model
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self._attn_implementation = attn_implementation

        # Ajustes hardcoded especificos para forzar a GPT-2 a comportarse como DTrOCR
        
        # desactiva la capa interna n_inner, para que GPT-2 use su valor predeterminado interno (generalmente 4 * hidden_size)
        self.n_inner = None
        
        # activa la escala de pesos de atencion (True) para evitar que los valores de atencion exploten numericamente antes de pasar por Softmax
        self.scale_attn_weights = True
        
        # desactiva el escalado inverso para mantener el estandar de calculo sin alterar la importancia de las capas profundas
        self.scale_attn_by_inverse_layer_idx = False
        
        # Se desactiva la reorganizacion de atencion para ahorrar tiempo de computo, ya que no es vital en este tipo de arquitectura
        self.reorder_and_upcast_attn = False        
        # desactiva la atención cruzada  para cumplir con la arquitectura DTrOCR solo decodificador, donde texto e imagen entran juntos y se leen a sí mismos, eliminando el uso de un codificador
        self.add_cross_attention = False
        
        # define la función de activación (gelu_new) para agregar no-linealidad a la red, ayudandola a aprender patrones complejos de una forma ligeramente mas fluida que el relu tradicional
        self.activation_function = "gelu_new"