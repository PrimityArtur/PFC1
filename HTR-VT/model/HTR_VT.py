import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from functools import partial

# permite que cada parte de la imagen (token) se compare con todos los demas partes al mismo tiempo para entender el contexto global
class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # verifica que la dimension total sea divisible entre el numero de cabezales para que se puedan repartir los datos matematicamente sin que sobren decimales
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        # divide la dimension total entre los cabezales para que cada cabezal procese una fraccion de la informacion de forma independiente
        head_dim = dim // num_heads
        # calcula un factor de escala (raiz cuadrada inversa) para que los valores de atencion no crezcan demasiado y desestabilicen el entrenamiento
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches
        
        # crea matrices de sesgo (bias) llenas de unos para que puedan ser usadas opcionalmente como mascaras de atencion direccional 
        self.bias = torch.ones(1, 1, self.num_patches, self.num_patches)
        self.back_bias = torch.triu(self.bias)
        self.forward_bias = torch.tril(self.bias)
        
        # crea una capa lineal que multiplicara la entrada por 3 para que genere simultaneamente las matrices Q (Consulta), K (Key) y V (Value)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # define un dropout para que apague conexiones aleatorias y evite el sobreajuste
        self.attn_drop = nn.Dropout(attn_drop)
        # crea otra capa lineal para que reensamble y proyecte la informacion combinada de vuelta a su dimension original
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # extrae el tamaño del lote (B), el numero de tokens (N) y los canales (C) para que la red sepa las dimensiones actuales
        B, N, C = x.shape
        
        # pasa la entrada por la capa qkv, se remodela y se transponen dimensiones para que la informacion quede dividida fisicamente en multiples cabezales
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # separa el tensor en Q, K y V para que puedan multiplicarse entre si
        q, k, v = qkv.unbind(0)

        # multiplica la consulta (q) por la clave transpuesta (k) y se escala para que la red calcule que tanta "atencion" debe prestarle un token a los demas
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # aplica la funcion softmax para que todas las puntuaciones de atencion se conviertan en probabilidades que sumen 1.0
        attn = attn.softmax(dim=-1)
        # aplica el dropout para que la matriz de atencion sea robusta ignorando ciertos enlaces
        attn = self.attn_drop(attn)

        # multiplica la matriz de probabilidades (attn) por los valores (v) para que cada token absorba la informacion de los tokens mas relevantes. Luego se remodela para que vuelva a ser un solo bloque continuo (B, N, C).
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # se proyecta el resultado final para que mezcle la informacion de todos los cabezales juntos
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# multiplica la salida de una capa por un valor aprendible (gamma). para que las redes muy profundas no colapsen 
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        # crea un parametro aprendible (gamma) cercana a cero para que la capa empiece sin afectar al modelo, y gane importancia a medida que entrena
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # Se multiplica la entrada x por gamma para que escale importancia
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
    
# bloque completo del Transformer. Contiene la atencion, la red neuronal (MLP) y las conexiones residuales
class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            num_patches,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.0,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        # define la primera normalizacion para que estabilice los datos antes de entrar a la atencion
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        # instancia el modulo Attention para que evalue el contexto global
        self.attn = Attention(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # instancia el LayerScale para que controle el impacto de la atencion
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # se añade un DropPath (Stochastic Depth) para que a veces apague este bloque entero durante el entrenamiento, obligando a los demas bloques a trabajar mas
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # define la segunda normalizacion para que prepare los datos hacia el MLP
        self.norm2 = norm_layer(dim, elementwise_affine=True)
        # crea una red Perceptron Multicapa (Mlp) para que procese independientemente la informacion recien enriquecida de cada token
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # conexion residual/ se suma la entrada (x) con el resultado de la atencion para que no se pierda la informacion visual original
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # segunda conexion residual/ suma el resultado actual con la salida del MLP para que se consoliden los datos procesados
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

# Como el Transformer evalua en paralelo, no sabe que pedazo de imagen va primero y cual despues. por lo que estas funciones usan ondas senoidales para inyectar un ubicaciones a cada token
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    # crean arreglos numericos para el alto y ancho para que representen las coordenadas de la cuadricula de la imagen
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    
    # combinan los ejes en una matriz bidimensional (meshgrid) para que cada token tenga una coordenada X e Y
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    
    # Se llama a la sub-funcion para que calcule las ondas basandose en la cuadricula
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # usa la mitad de la dimension de canales para que codifique la posicion horizontal (H)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    # usa la otra mitad para que codifique la posicion vertical (W)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    # se unen ambas mitades para que formen la etiqueta de ubicacion 
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    # crea un rango de frecuencias (omega) para que las ondas senoidales oscilen a diferentes velocidades
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    # hace un producto externo para que multiplique las posiciones por las frecuencias
    out = np.einsum('m,d->md', pos, omega)

    # aplica la funcion Seno y Coseno para que se generen patrones unicos 
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    # combinan los senos y cosenos para que el resultado final sea devuelto.
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

# funcion de normalizacion
class LayerNorm(nn.Module):
    def forward(self, x):
        # normaliza la capa sobre los tensores para que asegure que su media sea 0 y varianza 1
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

# junta la ResNet18, Transformer, enmascaramiento y el cabezal para producir las predicciones de las letras
class MaskedAutoencoderViT(nn.Module):
    def __init__(self,
                 nb_cls=80,
                 img_size=[512, 32] ,
                 patch_size=[8, 32],
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.layer_norm = LayerNorm()
        # instancia la ResNet18 para que extraiga los parches visuales iniciales
        self.patch_embed = resnet18.ResNet18(embed_dim)
        
        # calcula el tamaño de la cuadricula dividiendo el tamaño de la imagen entre el tamaño del parche para saber cuantos tokens van a salir
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # crea un token de mascara vacio (aprendible) para que reemplace a los parches ocultos y la red sepa donde tiene que adivinar
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # crea un espacio para guardar los Embeddings Posicionales para que se sumen a las imagenes luego. Se bloquea con requires_grad=False para que no se alteren
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        
        # apilan varios capas de Transformer en una lista para que el modelo adquiera profundidad de razonamiento
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, self.num_patches, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        # crea el cabezal final (Linear) para que traduzca las dimensiones complejas de los tensores a probabilidades de las letras del abecedario
        self.head = torch.nn.Linear(embed_dim, nb_cls)

        # llama a la funcion para inicializar los pesos 
        self.initialize_weights()

    def initialize_weights(self):
        # generan los valores de senos y cosenos y se inyectan en pos_embed para que la posicion este listo antes de entrenar
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # inicializa el mask_token con numeros aleatorios pequeños (distribucion normal) para que evite estancamientos iniciales
        torch.nn.init.normal_(self.mask_token, std=.02)

        # aplica la funcion _init_weights a todas las subcapas para que asegure correctamente los calculos
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # usa el inicializador de xavier_uniform para que los pesos de las capas lineales esten bien balanceados y no causen desvanecimiento de gradiente
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape  
        # crea una mascara de unos, para que represente que todo esta visible inicialmente
        mask = torch.ones(N, L, 1).to(x.device)
        
        # calcula la cantidad de pixeles a ocultar basandose en el mask_ratio para que deduzca la informacion visual faltante
        span_length = int(L * mask_ratio)
        # calcula cuantos bloques (spans) se van a hacer para que se oculten pedazos continuos y no letras aleatorias sueltas
        num_spans = span_length // max_span_length
        
        for i in range(num_spans):
            # elige un punto de inicio aleatorio
            idx = torch.randint(L - max_span_length, (1,))
            # convierte en ceros 0 ese tramo de la mascara para que marque que area debe esconderse
            mask[:,idx:idx + max_span_length,:] = 0
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        # llama a generate_span_mask para que decida que lugares ocultar
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        # hace una mezcla de si la mascara es 1, se queda la imagen original (x * 1). Si es 0, se inyecta el mask_token (mask_token * 1) para que el Transformer sepa que hay informacion borrada ahi
        x_masked = x * mask + (1 - mask) * self.mask_token
        return x_masked

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        #  normaliza la imagen de entrada 
        x = self.layer_norm(x)
        # extraen los mapas de caracteristicas visuales usando la ResNet18 para que los trazos de texto se vuelvan tensores 
        x = self.patch_embed(x)
        
        b, c, w, h = x.shape
        # aplana la imagen procesada de una matriz 2D a una secuencia lineal 1D para que el Transformer pueda leerla de izquierda a derecha
        x = x.view(b, c, -1).permute(0, 2, 1)
        
        # aplica la mascara solo si use_masking es True (en entrenamiento) para que obligue al modelo a inferir letras faltantes
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
            
        # suman las coordenadas de ubicacion (pos_embed) a las imagenes para que el modelo recuerde el orden original de las letras
        x = x + self.pos_embed
        
        # se pasa la secuencia por todos los bloques del Transformer para que analicen el contexto y deduzcan que dice la imagen
        for blk in self.blocks:
            x = blk(x)

        # normaliza por ultima vez 
        x = self.norm(x)
        
        # pasa por el cabezal MLP para que traduzca la matriz en las probabilidades del abecedario
        x = self.head(x)
        x = self.layer_norm(x)

        # Se retorna el resultado final al archivo train.py para que la funcion CTC pueda medir el error
        return x


# se llama desde otros archivos para no tener que escribir todas las configuraciones de la red 
def create_model(nb_cls, img_size, **kwargs):
    # Se instancia MaskedAutoencoderViT preconfigurado con 4 capas (depth=4) y 6 cabezales (num_heads=6), dim (768) para que coincida con el papaer
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 patch_size=(4, 64),
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    return model