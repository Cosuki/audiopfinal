# Dueto de Voces Automático

**Proyecto final de AudioDSP**, este repositorio contiene el codigo implementado para acondicionar y sincronizar 2 pistas de audio. Se usaron espesificamente la cancion 'Every breath you take' de The Police y un cover hecho por Emily Linge. 


Se encuentran los archivos:
- `pipelane_sincronizacion.ipynb`: Contiene la implementacion de las partes 1 hasta la 6, se obtiene desviacion de pitch con respecto a A4 (440Hz).
- `ajuste_timbre.ipynb`: Contiene una implementacion mas detallada de la parte 6 de ajustar el timbre del cover shifteado.
- `fonemas_polos.ipynb` y `fonemas.ipynb`: Contienen distintas implementaciones de la parte 7 de sincronizar las pistas a partir de caracteristicas timbricas.
- `music_synchronization.py`: Contiene el algoritmo para analizar y sincronizar 2 pistas de manera consisa.
- `utils/`: Notebooks de pruebas.


