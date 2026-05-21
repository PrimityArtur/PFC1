import itertools
import cv2
import numpy as np
from skimage import transform as stf
from numpy import random, floor
from PIL import Image, ImageOps
from cv2 import erode, dilate, normalize
from torchvision.transforms import RandomCrop
import math


# egrosa los trazos del texto, haciendo las letras mas anchas
class Dilation:
    def __init__(self, kernel, iterations):
        # Se crea una matriz llena de unos (1) en self.kernel usando np.ones. 
        # El parametro kernel define el tamaño de esta matriz y np.uint8 define el tipo de dato para que sea compatible con imagenes.
        self.kernel = np.ones(kernel, np.uint8)
        # Se guarda el numero de iteraciones
        self.iterations = iterations

    def __call__(self, x):
        # Se transforma la imagen de entrada x en una matriz numerica usando np.array(x)
        # luego se ejecuta la funcion dilate de OpenCV sobre esa matriz numerica, aplicando el self.kernel y repitiendolo self.iterations veces
        # para envolver el resultado en Image.fromarray para convertir los numeros en un archivo visual
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))

# disminuye los trazos del texto
class Erosion:
    def __init__(self, kernel, iterations):
        # Se inicializa una matriz de unos en la variable self.kernel de la misma forma que en Dilation.
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        # Se convierte la imagen x a un arreglo con np.array(x)
        # se aplica la funcion erode que reduce los bordes blancos/oscuros usando el self.kernel
        # y se retorna como imagen con Image.fromarray.
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))

# estira o encoge la imagen
class ElasticDistortion:
    def __init__(self, grid, magnitude, min_sep):
        # se asignan los valores de grid cuadricula en las variables self.grid_width y self.grid_height (columnas y filas)
        self.grid_width, self.grid_height = grid
        # aca los valores maximos de distorsion 
        self.xmagnitude, self.ymagnitude = magnitude
        # y las separaciones minimas permitidas entre puntos
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, x):
        # se extrae el ancho y el alto de la imagen x 
        w, h = x.size

        # dimensiones de la cuadricula
        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height
        
        # calcula el ancho y alto de cada celda y redondeado
        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        # aca se calcula el ancho y alto de la ultima celda restando al ancho total w,h on el espacio ocupado por todas las celdas anteriores
        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        # crea una lista bidimensional shift llena de coordenadas (0,0) que representara cuanto se va a mover cada punto de la cuadricula
        shift = [[(0, 0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                # Mediante las condiciones, se definen las coordenadas de origen [x1, y1, x2, y2] para cada celda de la cuadricula
                # luego se diferencia la ultima fila y ultima columna para usar width_of_last_square y height_of_last_square y que no sobren pixeles
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square, vertical_tile * height_of_square, width_of_last_square + (horizontal_tile * width_of_square), height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square, vertical_tile * height_of_square, width_of_square + (horizontal_tile * width_of_square), height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square, vertical_tile * height_of_square, width_of_last_square + (horizontal_tile * width_of_square), height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square, vertical_tile * height_of_square, width_of_square + (horizontal_tile * width_of_square), height_of_square + (height_of_square * vertical_tile)])

                # calcula el limite de movimiento horizontal (sm_h y sm_v) para que la distorsion no altere el cuadrante anterior
                sm_h = min(self.xmagnitude, width_of_square - (self.min_h_sep + shift[vertical_tile][horizontal_tile - 1][0])) if horizontal_tile > 0 else self.xmagnitude
                sm_v = min(self.ymagnitude, height_of_square - (self.min_v_sep + shift[vertical_tile - 1][horizontal_tile][1])) if vertical_tile > 0 else self.ymagnitude

                # Se genera un numero aleatorio para mover el punto en el eje X y Y entre el limite negativo y la magnitud maxima
                # y se guardan estos desplazamientos en la matriz shift
                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        # Se aplana la matriz bidimensional shift en una sola lista
        shift = list(itertools.chain.from_iterable(shift))

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        # Se identifican los indices de la ultima fila
        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            # Se toman las 2 coordenadas (esquina superior izquierda e inferior derecha) de dimensions y se expanden a 4 coordenadas (8 puntos) para formar el poligono completo
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                # Se enlazan los vertices vecinos para saber que esquinas comparten movimiento
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id, (a, b, c, d) in enumerate(polygon_indices):
            # Se extrae el desplazamiento para el indice actual id.
            dx = shift[id][0]
            dy = shift[id][1]

            # Se modifica la esquina inferior derecha del poligono a sumandole dx y dy.
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4]

            # modifica la esquina inferior izquierda del poligono b
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1, x2 + dx, y2 + dy, x3, y3, x4, y4]

            # modifica la esquina superior derecha del poligono c
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1, x2, y2, x3, y3, x4 + dx, y4 + dy]

            # modifica la esquina superior izquierda del poligono d
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy, x2, y2, x3, y3, x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            # Se empaqueta el cuadrante original dimensions[i] con su version deformada polygons[i] 
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh
        
        # Se llama a la funcion interna de la libreria PIL x.transform, se le pasa la malla generada y se reconstruye la imagen usando interpolacion Image.BICUBIC.
        return x.transform(x.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)

# Inclina la imagen
class RandomTransform:
    def __init__(self, val):

        self.val = val

    def __call__(self, x):
        w, h = x.size
        
        # Decide aleatoriamente si distorsionar el ancho o el alto
        # Se evalua random.randint(0,2). Si es 0, la variable dw toma el valor de self.val y dh es 0. Si no, es al reves. Esto decide si se deforma el eje X o el Y.
        dw, dh = (self.val, 0) if random.randint(0, 2) == 0 else (0, self.val)
        def rd(d):
            return random.uniform(-d, d)
        def fd(d):
            return random.uniform(-dw, d)

        # Se generan desplazamientos aleatorios para las esquinas superior-izquierda (tl), inferior-izquierda (bl), superior-derecha (tr) e inferior-derecha (br) a la iamgen
        tl_top = rd(dh)
        tl_left = fd(dw)
        bl_bottom = rd(dh)
        bl_left = fd(dw)
        tr_top = rd(dh)
        tr_right = fd(min(w * 3 / 4 - tl_left, dw))
        br_bottom = rd(dh)
        br_right = fd(min(w * 3 / 4 - bl_left, dw))

        # Se crea un objeto ProjectiveTransform 
        tform = stf.ProjectiveTransform()
        # Se calcula la matriz de perspectiva dandole las nuevas coordenadas desordenadas (primer array) frente a las coordenadas perfectas de un rectangulo (segundo array).
        tform.estimate(np.array((        
            (tl_left, tl_top),
            (bl_left, h - bl_bottom),
            (w - br_right, h - br_bottom),
            (w - tr_right, tr_top)
        )), np.array((
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        )))

        # matriz de transformacion para inclinar la imagen en las esquinas base 
        corners = np.array([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ])
        
        # Conserva el tamaño calculando las nuevas coordenadas de salida
        # proyectan las esquinas base a traves de la transformacion inversa para saber donde terminarian
        corners = tform.inverse(corners)
        # ahora buscan los valores minimos y maximos de X y de Y para encontrar los uevos limites de la imagen deformada
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        # Se calcula el nuevo numero de filas y columnas
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        # Se redondean estos valores 
        output_shape = np.around((out_rows, out_cols))

        # Se crea una transformacion de traslacion en tform4 para mover la imagen de vuelta al origen usando minc y minr
        translation = (minc, minr)
        tform4 = stf.SimilarityTransform(translation=translation)
        # Se suma la traslacion a la deformacion principal en tform
        tform = tform4 + tform
        # Se normaliza la matriz de transformacion
        tform.params /= tform.params[2, 2]

        # Se aplica la deformacion matematica tform a la matriz np.array(x) usando la funcion warp. El fondo vacio se rellena con color blanco 
        x = stf.warp(np.array(x), tform, output_shape=output_shape, cval=255, preserve_range=True)
        # Se fuerza a la imagen resultante x a volver a sus dimensiones originales h, w.
        x = stf.resize(x, (h, w), preserve_range=True).astype(np.uint8)

        # Se devuelve el arreglo reconstruido como un archivo visual
        return Image.fromarray(x)


# Invierte los colores blanco y negro
class SignFlipping:
    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)

# Agranda o reduce la imagen
class DPIAdjusting:
    def __init__(self, factor, preserve_ratio):
        self.factor = factor
    def __call__(self, x):
        w, h = x.size
        # se aplica la funcion resize con las medidas por self.factor 
        # Image.BILINEAR suaviza los pixeles durante el cambio de tamaño
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)


# agrega puntos a la imagen.
class GaussianNoise:
    def __init__(self, std):
        # la desviacion estandar (la fuerza q tiene el ruido) 
        self.std = std

    def __call__(self, x):
        x_np = np.array(x)
        mean, std = np.mean(x_np), np.std(x_np)
        std = math.copysign(max(abs(std), 0.000001), std)
        min_, max_ = np.min(x_np,), np.max(x_np)
        # Genera el ruido
        normal_noise = np.random.randn(*x_np.shape)
        if len(x_np.shape) == 3 and x_np.shape[2] == 3 and np.all(x_np[:, :, 0] == x_np[:, :, 1]) and np.all(x_np[:, :, 0] == x_np[:, :, 2]):
            normal_noise[:, :, 1] = normal_noise[:, :, 2] = normal_noise[:, :, 0]
        # aplicar ruido y normaliza para evitar colores saturados
        x_np = ((x_np-mean)/std + normal_noise*self.std) * std + mean
        x_np = normalize(x_np, x_np, max_, min_, cv2.NORM_MINMAX)

        return Image.fromarray(x_np.astype(np.uint8))

# Resalta los bordes y el contraste de los colores.
class Sharpen:
    def __init__(self, alpha, strength):
        self.alpha = alpha
        self.strength = strength

    def __call__(self, x):
        x_np = np.array(x)
        id_matrix = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]]
                             )
        # Matriz para resaltar bordes
        effect_matrix = np.array([[1, 1, 1],
                                  [1, -(8+self.strength), 1],
                                  [1, 1, 1]]
                                 )
        # Fusiona la imagen original con el efecto de nitidez usando el factor alpha
        kernel = (1 - self.alpha) * id_matrix - self.alpha * effect_matrix
        kernel = np.expand_dims(kernel, axis=2)
        kernel = np.concatenate([kernel, kernel, kernel], axis=2)
        sharpened = cv2.filter2D(x_np, -1, kernel=kernel[:, :, 0])
        return Image.fromarray(sharpened.astype(np.uint8))

# Recorta un area de la imagen y luego lo agranda al tamaño inicial.
class ZoomRatio:
    def __init__(self, ratio_h, ratio_w, keep_dim=True):
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.keep_dim = keep_dim

    def __call__(self, x):
        w, h = x.size
        # Se toma la funcion RandomCrop de torchvision y se le pasa a la imagen x para el recorte
        x = RandomCrop((int(h * self.ratio_h), int(w * self.ratio_w)))(x)
        # si self.keep_dim es True, la imagen recortada x se vuelve a estirar a sus medidas originales w y h
        if self.keep_dim:
            x = x.resize((w, h), Image.BILINEAR)
        return x

# Busca filas horizontales que solo tengan color blanco, las borra
class Tightening:
    def __init__(self, color=255, remove_proba=0.75):
        self.color = color
        self.remove_proba = remove_proba

    def __call__(self, x):
        x_np = np.array(x)
        # Identifica cuales filas horizontales blancas
        interline_indices = [np.all(line == 255) for line in x_np]
        # de forma aleatoria decidir cuales lineas borrar
        indices_to_removed = np.logical_and(np.random.choice([True, False], size=len(x_np), replace=True, p=[self.remove_proba, 1-self.remove_proba]), interline_indices)
        # Se filtra la imagen original x_np usando np.logical_not. osea se conservan todas las lineas de pixeles menos las marcadas en indices_to_removed.
        new_x = x_np[np.logical_not(indices_to_removed)]
        return Image.fromarray(new_x.astype(np.uint8))