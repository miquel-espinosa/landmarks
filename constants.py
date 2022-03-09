BASE_DIR = "/home/miquel/Documents/MASTER/Robots/Landmarks"
IMG_DIR = BASE_DIR + "/new-images"
# IMG_DIR = BASE_DIR + "/Capturas/Imágenes"
TEST_DIR = BASE_DIR + "/Capturas/Test"

DELTA = 0.02 # 1nn
Q_VALUE = 10 # qnn

# values: qnn, 1nn
ALGORITMO = "1nn" 

# Para determinar si quiero generar video o no
VIDEO = True

# Si quiero calcular la accuracy
ACCURACY = False

# Si quiero generar los hist medios por nodo
MEAN_HIST = False

# nodes = {
#     0: "Nodo1",
#     1: "Nodo2",
#     2: "Nodo3",
#     3: "Nodo3TG",
#     4: "Nodo4",
#     5: "Nodo5A",
#     6: "Nodo5B",
#     7: "Nodo6",
#     8: "Nodo7",
# }

nodes = {
    0: "Nodo1",
    1: "Nodo2",
    2: "Nodo3",
    3: "Nodo4",
    4: "Nodo5A",
    5: "Nodo5B",
    6: "Nodo6",
    7: "Nodo7",
}