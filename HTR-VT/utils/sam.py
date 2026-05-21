import torch

# Sharpness-Aware Minimization
# optimizador. En lugar de actualizar los pesos una sola vez, Sam da un paso en falso hacia el peor error posible, calcula la pendiente ahi, regresa a su posicion original, y usa esa informacion para dar un paso seguro
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        # usa assert para verificar que el radio de busqueda (rho) sea positivo, para que la matematica de la perturbacion no se invierta ni rompa el algoritmo
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        # agrupan los parametros base como 'rho' y 'adaptive' en un diccionario defaults para que PyTorch pueda gestionarlos internamente
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)

        super(SAM, self).__init__(params, defaults)

        # inicializa el optimizador (base_optimizer, que en train.py es AdamW) pasandole los parametros para que el sea quien haga los ajustes numericos finales
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # enlazan los grupos de parametros (pesos de las neuronas) para que SAM y AdamW apunten exactamente al mismo espacio de memoria
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # calcula la magnitud total de los gradientes (grad_norm) para saber que tan inclinada esta la pendiente del error en la posicion actual
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            # calcula la escala de perturbacion dividiendo rho entre la norma, sumando 1e-12 para que nunca se divida por cero y el programa no falle
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                # salta la iteracion si el parametro no tiene gradiente (capas congeladas) para no perder tiempo de procesamiento
                if p.grad is None: continue
                
                # crea un clon de los pesos actuales y se guardan en "old_p" para que el modelo recuerde su posicion original y pueda regresar 
                self.state[p]["old_p"] = p.data.clone()
                
                # calcula el vector de perturbacion 'e_w' multiplicando el gradiente por la escala calculada, para saber en que direccion exacta el error sube mas rapido
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                
                # suma la perturbacion 'e_w' directamente a los pesos del modelo 'p' para que la red intencionalmente trepando al pico de error mas cercano
                p.add_(e_w)

        # limpian los gradientes viejos si zero_grad es True para que la red este lista para un nuevo calculo forward/backward desde este punto
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # Se sobrescriben los pesos actuales con los que guardamos en "old_p" para que la red regrese a su estado original (baje del pico de error).
                p.data = self.state[p]["old_p"]  

        # Se manda a llamar al optimizador base (AdamW) para que aplique la actualizacion de los pesos usando los gradientes de la posicion obtenidas.
        self.base_optimizer.step()  

        # limpian los gradientes para que la red pueda con la siguiente iteracion del bucle de entrenamiento
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # exige que la funcion step reciba un 'closure' (que es la funcion que calcula el forward y el backward) para que SAM pueda ejecutar la red dos veces
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # habilita temporalmente el calculo de gradientes dentro del closure para que PyTorch no congele el aprendizaje
        closure = torch.enable_grad()(closure) 

        # se da el paso hacia el peor error, guardando el estado original y limpiando la pizarra
        self.first_step(zero_grad=True)
        # vuelve a correr la red neuronal (closure) desde ese punto malo para que calcule los gradientes del "peor escenario"
        closure()
        # restaura el modelo original y se aplican los ajustes seguros basandose en la advertencia del paso 2
        self.second_step()

    def _grad_norm(self):
        # identifica en que procesador o tarjeta grafica (device) estan los tensores para que PyTorch no se confunda al sumar datos
        shared_device = self.param_groups[0]["params"][0].device  
        
        # apilan todos los gradientes de todas las neuronas, se saca su norma L2 (teorema de pitagoras ) y se suma todo para que la red resuma su estado de inclinacion en un solo valor matricial
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        # devuelve esta magnitud unica para que first_step sepa que tanto puede alterar los pesos
        return norm

    def load_state_dict(self, state_dict):
        # restaura el estado general guardado en los archivos .pth para que puedas retomar el entrenamiento desde donde se quedo
        super().load_state_dict(state_dict)
        # resincronizan los grupos de parametros con el optimizador base para que no haya desajustes en la memoria RAM tras cargar el modelo
        self.base_optimizer.param_groups = self.param_groups