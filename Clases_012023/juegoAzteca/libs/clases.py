class aguila:
    fuerza = 50
    rapidez = 70
    astucia = 90
    def __init__(self, nombre, ):
        self.nombre = nombre
        self.salud = 100

    def espia(self):
        assert self.salud > 0, "Caballero muerto"
        print("Averiguando informacion")

    def atacar(self, enemigo):
        assert self.salud > 0, "Caballero muerto"
        enemigo.salud = enemigo.salud - int(10 * (self.fuerza / 100))
        print("salud de %s %d"%(enemigo.nombre, enemigo.salud))

    def rezar(self, dios):
        assert self.salud > 0, "Caballero muerto"
        self.__sanar__()

    def __sanar__(self):
        if self.salud < 90:
            self.salud += 10
        else:
            print("Salud al maximo")


class jaguar(aguila):
    fuerza=80
    rapidez=80
    astucia=50
    def __init__(self, nombre):
        self.nombre = nombre
        self.salud = 100

    def correr(self, distancia):
        assert self.salud > 0, "jaguar muerto"
        print("Corriendo ...")
        for i in range(distancia//1000):
            self.salud -= 1
        print("Salud %d"%self.salud)


if __name__ == "__main__":
    print("Hola")
    itzel = jaguar("Itzel")
