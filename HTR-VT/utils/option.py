import argparse

# configura el diccionario de variables ingresadas en la consola para train.py y test.py pueden usar para configurar la red
def get_args_parser():
    parser = argparse.ArgumentParser(description='HTR-VT',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ENTRENAMIENTO
    #para que el programa sepa en que carpeta del disco duro debe guardar los resultados y los pesos entrenados (.pth)
    parser.add_argument('--out-dir', type=str, default='./output', help='output directory')
    
    # (Batch Size) para que la tarjeta grafica sepa exactamente cuantas imagenes de entrenamiento debe cargar y procesar simultaneamente en cada paso
    parser.add_argument('--train-bs', default=8, type=int, help='train batch size')
    
    # para que la tarjeta grafica sepa cuantas imagenes de validacion procesar a la vez durante las pausas de evaluacion
    parser.add_argument('--val-bs', default=1, type=int, help='validation batch size')
    
    # para que el procesador CPU asigne multiples hilos de trabajo paralelos y cargue las imagenes mas rapido desde el disco a la RAM
    parser.add_argument('--num-workers', default=0, type=int, help='nb of workers')
    
    # para que el bucle de entrenamiento sepa cada cuantos pasos debe detenerse temporalmente para evaluar el modelo y guardar su progreso
    parser.add_argument('--eval-iter', default=1000, type=int, help='nb of iterations to run evaluation')
    
    # para que el sistema sepa cual es el limite maximo de pasos matematicos antes de dar por terminado todo el entrenamiento
    parser.add_argument('--total-iter', default=100000, type=int, help='nb of total iterations for training')
    
    # para que el optimizador sepa durante cuantas iteraciones debe mantener una velocidad de aprendizaje baja, evitando que la red colapse al principio
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='nb of iterations for warm-up')
    
    # para que la consola sepa cada cuantos pasos debe imprimir el texto con el estado actual del error (loss), para monitorearlo
    parser.add_argument('--print-iter', default=100, type=int, help='nb of total iterations to print information')
    
    # (Learning Rate) para que el optimizador sepa cual es el tamaño maximo de los "pasos" que puede dar al actualizar los pesos de las neuronas
    parser.add_argument('--max-lr', default=1e-3, type=float, help='learning rate')
    
    # para que actue como un castigo matematico sobre los pesos muy grandes, forzando a la red a mantenerse simple y evitar memorizar los datos (sobreajuste)
    parser.add_argument('--weight-decay', default=5e-1, type=float, help='weight decay')
    
    # booleano para que el script decida si enviar los graficos de progreso a la plataforma online Weights & Biases o usar TensorBoard localmente
    parser.add_argument('--use-wandb', action='store_true', default=False, help = 'wheteher use wandb, otherwise use tensorboard')
    
    # para que el programa cree una subcarpeta con este nombre especifico y no mezcle los archivos de diferentes experimentos
    parser.add_argument('--exp-name',type=str, default='IAM_HTR_ORIGAMI_NET', help='experimental name (save dir will be out_dir + exp_name)')
    
    # para que los generadores de numeros aleatorios empiecen siempre desde el mismo punto, asegurando que los experimentos sean reproducibles 
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')

    # TRANSFORMER
    # pasando dos valores para que la CNN y el DataLoader sepan a que resolucion (ancho y alto) deben aplastar todas las imagenes entrantes
    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+', help='image size')
    
    # para que el mecanismo de atencion del Transformer sepa que porcentaje de conexiones internas debe apagar (dropout) para ser mas robusto
    parser.add_argument('--attn-mask-ratio', default=0., type=float, help='attention drop_key mask ratio')
    
    # para que el modelo sepa en que dimensiones recortar la imagen antes de convertirla en tokens para el Transformer
    parser.add_argument('--patch-size', default=[4, 32], type=int, nargs='+', help='patch size')
    
    # para que la tecnica Span Mask sepa que porcentaje total de la imagen debe borrar (volver bloques negros) durante el entrenamiento
    parser.add_argument('--mask-ratio', default=0.3, type=float, help='mask ratio')
    
    # para que escale matematicamente la similitud del clasificador, controlando que tan "segura" debe estar la red al predecir una letra
    parser.add_argument('--cos-temp', default=8, type=int, help='cosine similarity classifier temperature')
    
    # para que la tecnica Span Mask sepa el tamaño maximo (en tokens continuos) que puede tener un solo bloque negro, forzando a la red a inferir trazos largos perdidos
    parser.add_argument('--max-span-length', default=4, type=int, help='max mask length')
    
    # para que establezca una distancia minima de separacion entre multiples bloques negros, evitando tapar toda la palabra junta
    parser.add_argument('--spacing', default=0, type=int, help='the spacing between two span masks')
    
    # para que la transformacion aleatoria sepa cual es el limite de distorsion al proyectar e inclinar la imagen en 3D
    parser.add_argument('--proj', default=8, type=float, help='projection value')

    # CONFIGURACIONES DE AUMENTO DE DATOS SOBRE IAM
    # limites maximos y minimos para que el archivo transform.py sepa exactamente hasta que punto puede alterar una imagen sin destruirla por completo
    parser.add_argument('--dpi-min-factor', default=0.5,type=float)
    parser.add_argument('--dpi-max-factor', default=1.5, type=float)
    parser.add_argument('--perspective-low', default=0., type=float)
    parser.add_argument('--perspective-high', default=0.4, type=float)
    parser.add_argument('--elastic-distortion-min-kernel-size', default=3, type=int)
    parser.add_argument('--elastic-distortion-max-kernel-size', default=3, type=int)
    parser.add_argument('--elastic_distortion-max-magnitude', default=20, type=int)
    parser.add_argument('--elastic-distortion-min-alpha', default=0.5, type=float)
    parser.add_argument('--elastic-distortion-max-alpha', default=1, type=float)
    parser.add_argument('--elastic-distortion-min-sigma', default=1, type=int)
    parser.add_argument('--elastic-distortion-max-sigma', default=10, type=int)
    
    # para que el filtro de dilatacion/erosion sepa cual es el tamaño maximo del pincel al engrosar o adelgazar las letras
    parser.add_argument('--dila-ero-max-kernel', default=3, type=int )
    
    # limitan las alteraciones de color para que el filtro Jitter sepa cuanto puede cambiar el contraste, brillo y saturacion de los pixeles
    parser.add_argument('--jitter-contrast', default=0.4, type=float)
    parser.add_argument('--jitter-brightness', default=0.4, type=float )
    parser.add_argument('--jitter-saturation', default=0.4, type=float )
    parser.add_argument('--jitter-hue', default=0.2, type=float)

    # filtros graficos (difuminado, enfoque, acercamiento) para que el codigo de transformaciones limite su intensidad
    parser.add_argument('--dila-ero-iter', default=1, type=int, help='nb of iterations for dilation and erosion kernel')
    parser.add_argument('--blur-min-kernel', default=3, type=int)
    parser.add_argument('--blur-max-kernel', default=5, type=int)
    parser.add_argument('--blur-min-sigma', default=3, type=int)
    parser.add_argument('--blur-max-sigma', default=5, type=int)
    parser.add_argument('--sharpen-min-alpha', default=0, type=int)
    parser.add_argument('--sharpen-max-alpha', default=1, type=int)
    parser.add_argument('--sharpen-min-strength', default=0, type=int)
    parser.add_argument('--sharpen-max-strength', default=1, type=int)
    parser.add_argument('--zoom-min-h', default=0.8, type=float)
    parser.add_argument('--zoom-max-h', default=1, type=float)
    parser.add_argument('--zoom-min-w', default=0.99, type=float)
    parser.add_argument('--zoom-max-w', default=1, type=float)
    
    # para que asigne una probabilidad central 50% decidiendo si se aplican o no ciertos efectos a cada imagen especifica
    parser.add_argument('--proba', default=0.5, type=float)

    # CONFIGURACIONES DEL MODELO EMA Y PERDIDAS 
    # para que la red "Sombra" sepa con que factor matematico debe actualizar y suavizar sus pesos basandose en la red principal
    parser.add_argument('--ema-decay', default=0.9999, type=float, help='Exponential Moving Average (EMA) decay')
    
    # para que si se usa una funcion de perdida secundaria (como KLD), sepa que tanto peso darle respecto a la perdida CTC principal
    parser.add_argument('--alpha', default=0, type=float, help='kld loss ratio')


    # Al escribir "IAM" al final del comando, se carga todas las rutas y el alfabeto especifico 
    # Se crea un contenedor de subcomandos en la variable para elegir entre diferentes bases de datos al ejecutar
    subparsers = parser.add_subparsers(title="dataset setting", dest="subcommand")

    # CONFIGURACION PARA IAM
    # Se crea el subcomando 'IAM' para que asigne configuraciones exclusivas al usar el conjunto de datos IAM.
    IAM = subparsers.add_parser("IAM",
                                description='Dataset parser for training on IAM',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on IAM")

    # definen las rutas por defecto de IAM para que el DataLoader sepa exactamente en que carpetas buscar las listas de entrenamiento, validacion y las imagenes
    IAM.add_argument('--train-data-list', type=str, default='./data/iam/train.ln', help='train data list (gc file)(ln file)')
    IAM.add_argument('--data-path', type=str, default='./data/iam/lines/', help='train data list')
    IAM.add_argument('--val-data-list', type=str, default='./data/iam/val.ln', help='val data list')
    IAM.add_argument('--test-data-list', type=str, default='./data/iam/test.ln', help='test data list')
    
    # 80 para que la ultima capa del modelo se construya con exactamente 80 neuronas de salida (79 caracteres del alfabeto IAM + 1 token blanco de CTC).
    IAM.add_argument('--nb-cls', default=80, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')

    # CONFIGURACION PARA READ2016 
    READ = subparsers.add_parser("READ",
                                 description='Dataset parser for training on READ',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 help="Dataset parser for training on READ")

    READ.add_argument('--train-data-list', type=str, default='./data/read2016/train.ln', help='train data list (gc file)(ln file)')
    READ.add_argument('--data-path', type=str, default='./data/read2016/lines/', help='train data list')
    READ.add_argument('--val-data-list', type=str, default='./data/read2016/val.ln', help='val data list')
    READ.add_argument('--test-data-list', type=str, default='./data/read2016/test.ln', help='test data list')
    READ.add_argument('--nb-cls', default=90, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')

    # CONFIGURACION PARA LAM
    LAM = subparsers.add_parser("LAM",
                                description='Dataset parser for training on LAM',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on READ")

    LAM.add_argument('--train-data-list', type=str, default='./data/LAM/train.ln', help='train data list (gc file)(ln file)')
    LAM.add_argument('--data-path', type=str, default='./data/LAM/lines/', help='train data list')
    LAM.add_argument('--val-data-list', type=str, default='./data/LAM/val.ln', help='val data list')
    LAM.add_argument('--test-data-list', type=str, default='./data/LAM/test.ln', help='test data list')
    
    LAM.add_argument('--nb-cls', default=90, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')

    return parser.parse_args()