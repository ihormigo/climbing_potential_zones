

import sys # Esta librería nos permite acceder a la configuración del sistema
import os # Con esta librería manipularemos rutas (unión, normalización de rutas,...)

import numpy as np # Se emplea para el cálculo con matrices
import geopandas as gpd # Es la librería que se usa para abrir y procesar archivos vectoriales (.shp)
import rasterio         # Es una de las librerías que se usan para procesar archivos raster.
from osgeo import gdal  # Se emplea para leer y procesar datos vectoriales y raster


class AnalisisRelieve:
    
    def __init__(self):
        # Se inicializan los siguientes parámetros para simplificar el paso de estos entre funciones. 
        self.crs = 'epsg:25830'

        # Los siguientes parametros inicializan las coordenadas de la caja (bbox) de nuestra zona de estudio.
        self.xmin = None 
        self.xmax = None 
        self.ymin = None 
        self.ymax = None   
        
        self.ancho = None
        self.alto = None
        self.resolucion = 2
        self.bandas = 1 # Número de bandas del archivo raster
        
    #-----------------------------------------------------------------------------------------------------------------------------------

    # La función 'abrir_array_raster' se emplea para abrir un archivo raster, ya sea para su procesado (reproyectar= False)
    # o para su visualización en otras funciones (reproyectar=True, en este caso es necesario reproyectar el raster al epsg:4326 para 
    # poder usar OSM como mapa base). 
    # El argumento 'metodo' indica el método de interpolación que se usará en la reproyección. 
    # Hay dos opciones: 'media' y por defecto la interpolación bilineal. Este argumento se tendrá en cuenta en casos como la energía 
    # del relieve donde la mayoría de los valores del array son NaN o -9999, lo que afecta al resultado de la reproyección, y por tanto,
    # a su visualización en el mapa. 
    def abrir_array_raster (self, ruta_raster, reproyectar = False, metodo=''):
        # El módulo 'rasterio' se emplea para abrir y procesar archivos raster:
        #  - La función 'open' como indica su nombre nos permite abrir un archivo raster.
        #  - La función 'band' nos permite acceder a una banda concreta del archivo raster (en el caso de un raster multidimensional 
        #    con más de 3 bandas).
        from rasterio import open, band  

        # La clase 'MemoryFile' dentro del submódulo 'rasterio.io' se emplea para crear un archivo temporal raster (en memoria). 
        from rasterio.io import MemoryFile
        
        # Dentro del submódulo 'rasterio.warp' se han empleado las siguientes funciones/clases:
        #  - calculate_default_transform: calcula los parámetros 'transformación', ancho y alto, necesarios para reproyectar 
        #                                 un raster a otro sistema de coordenadas (CRS).
        #  - reproject: se usa para reproyectar archivos raster.
        #  - Resampling: define el método de resampleo que se usará en la reproyección: nearest, bilinear, cubic, average.
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        # Esta es la opción que se usará para abrir un archivo raster que se vaya a procesar:
        if reproyectar is False: 
            with open(ruta_raster) as raster:
                array = raster.read(1) # En el caso de arrays raster multidimensionales/banda (3 o más dimensiones) sólo abre 
                                       # la primera banda (2D), ya que la librería 'folium' (para crear mapas interactivos) 
                                       # no admite imágenes con más de 3 bandas o dimensiones (sino es para establecer la
                                       # transparencia como se hace, por ejemplo, en el caso de 'energia del relieve').
                
                perfil = raster.profile # Este es el perfil que contiene los metadatos del raster.
                perfil.update(dtype='float32', nodata=-9999) # Una vez accedemos al perfil podemos actualizarlo con 'update'. 
                                                             # En este caso se ha definido el tipo de datos como flotante de 32 bits
                                                             # y el valor -9999 como 'nodata' 
                
                # Comprobamos si tiene CRS; si no, asignamos EPSG:25830 (definido en la inicialización como 'self.crs'):
                if raster.crs is None:
                    # Para definir el sistema de coordenadas se emplea la clase 'CRS' del submódulo 'rasterio.crs'
                    from rasterio.crs import CRS
                    perfil.update(crs=self.crs)
                
                raster_bounds = raster.bounds # A través de la propiedad 'bounds' del raster cargado podemos acceder 
                                              # a las coordenadas de los límites que lo definen (bbox).
                
                # Inicializamos las coordenadas del bbox, de forma que se puedan usar como parámetros en otras funciones, 
                # por ejemplo, cuando se cargan varios archivos en un mapa (de esta forma nos aseguramos que el bbox es el 
                # mismo para todas las capas).
                self.xmin = raster_bounds.left 
                self.ymin = raster_bounds.bottom
                self.xmax = raster_bounds.right
                self.ymax = raster_bounds.top
                # Para extraer la resolución del raster empleamos el atributo 'a' (tamaño del píxel X) de la propiedad 'transform'. 
                # Los datos raster de este proyecto tienen una resolución de 2m. 
                self.resolucion = raster.transform.a

        
        else: # Para reproyectar sólo los archivos raster que se representen en el mapa (OSM tiene crs 4326).
            with open(ruta_raster) as raster:
                crs_destino = "EPSG:4326"
                raster_bounds  = raster.bounds # Para extraer las coordenadas del bbox del raster.

                # Antes de reproyectar es necesario calcular la transformación que se aplicará junto con el alto y el ancho
                # tras la proyección:
                transform, width, height = calculate_default_transform (raster.crs, crs_destino, 
                                                                        raster.width, raster.height, 
                                                                        raster_bounds.left, raster_bounds.bottom, 
                                                                        raster_bounds.right, raster_bounds.top)
                # Creamos una copia del perfil original para modificarlo con los parámetros de la nueva transformación sin alterar 
                # el original (lo que puede provocar errores en la ejecución cuando se cargan varios archivos raster en memoria, como 
                # ocurre en la visualización).
                perfil = raster.profile.copy() 
                # Actualizamos el perfil del raster a reproyectar con los parámetros del nuevo crs. 
                perfil.update({
                    'crs': crs_destino,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'dtype': 'float32',
                    'nodata': -9999})
                
                # Recalculamos los límites del bbox a epsg:4326 a partir del transform reproyectado anterior:
                # 'a' = transform.a: ancho del píxel (es necesario hacer una conversión de metros (epsg:25830) a grados (epsg:4326), 
                #                                                  por eso tenemos que recalcular el ancho y el alto del píxel).
                # 'e' = transform.e: alto del píxel.
                # 'c' = transform.c: coordenada X de la esquina superior izquierda del raster (equivale a 'xmin')
                # 'f' = transform.f: coordenada Y del esquina superior izquierda del raster (equivale a 'ymax')
                
                xmin_4326 = transform.c
                ymax_4326 = transform.f
                xmax_4326 = xmin_4326 + width * transform.a # xmax = xmin + ancho del raster (nº de píxeles) x ancho del píxel (en el nuevo crs) 
                ymin_4326 = ymax_4326 + height * transform.e # ymin = ymax + alto del raster (nº de píxeles) x alto del píxel (en el nuevo crs) 
                                                             # Como partimos de la esquina superior izquierda los valores de las coordenadas en Y
                                                             # van decreciendo desde 'ymin' hasta 'ymax'.
                
                # Inicializamos las coordenadas del bbox con el nuevo crs:
                self.xmin = xmin_4326
                self.xmax = xmax_4326
                self.ymin = ymin_4326
                self.ymax = ymax_4326
                
                # Para mejorar el procesamiento de la función cargaremos el raster en memoria, ya que al final no la vamos a guardar 
                # en disco, sino que la devolveremos como un archivo array en memoria. De lo contrario sería necesario hacer el proceso
                # de reproyección y guardado en dos funciones/pasos distintos.
                # Con la función 'MemoryFile' del submódulo 'rasterio.io' podemos crear un "espacio" vacío en memoria para añadir  
                # posteriormente el raster reproyectado.
                raster_memoria = MemoryFile()

                # Definimos los métodos de interpolación que se usarán en la reproyección:
                if metodo == 'media':
                    metodo_resam = Resampling.average # Para evitar perder o alterar los datos de energía del relieve en el mapa (pocos 
                                                      # datos y muchos valores próximos a cero).  
                                                      
                else: # En el resto de casos (datos continuos) se emplea el métodos de interpolación bilinear para el resampleado:
                    metodo_resam = Resampling.bilinear
                    
                # Para empezar, abrimos el contenedor creado en memoria anteriormente como un raster vacío con 'open' y añadimos 
                # los pares clave-valor del diccionario que contiene el perfil (metadatos del raster). 
                with raster_memoria.open(**perfil) as raster_mem:
                    #for i in range(1, raster.count + 1):
                # Reproyectamos con la función 'reproject' del submódulo 'rasterio.warp'
                    reproject(
                        source = band(raster,1),               # Es necesario extraer la primera banda con la función 'band' porque no se 
                                                               # trata de un array (en este caso, 'else', no hemos leido el raster con 'read').
                        destination = band(raster_mem,1),      # Guardamos los datos en el raster vacío creado en memoria (de nuevo tenemos 
                                                               # que indicar el número de banda).
                        src_transform = raster.transform,      # 'transform' del raster original
                        src_crs = raster.crs,                  
                        dst_transform = transform,             # 'transform' del raster de destino calculado previamente
                        dst_crs = crs_destino,
                        src_nodata = -9999,                    # Valores 'nodata' de los datos de entrada. 
                        dst_nodata = -9999,                    # Valores 'nodata' de los datos de salida.
                        # Se usará la media como método de resampleo de la energía del relieve para que:
                        # - No se amplifiquen los outliers (como ocurre con los métodos bilineal y cúbico).
                        # - No se pierdan datos. En este caso en particular, la distribución de los datos presenta una 
                        # asimetría negativa hacia valores próximos a cero, por lo que al resamplear con el método bilineal 
                        # o cúbico se pierden la mayoría de los datos (NaN).
                        resampling = metodo_resam)
                    array = raster_mem.read(1) # Por defecto la reproyección devuelve un array de tres dimensiones, donde el primer 
                                               # valor representa el número de banda, por lo que es necesario explicitar la primera
                                               # aunque sólo usemos datos raster monobanda.
                    
        return array # Finalmente la función devuelve un raster como array, de esta forma podemos usarlo 
                     # como argumento de entrada en otras funciones.

    
    #-----------------------------------------------------------------------------------------------------------------------------------
    
    # La función 'guardar_array' parte de datos array en memoria para generar un archivo .tif en disco.
    def guardar_array(self, raster, ruta_salida, resolucion=2, xmin= None, ymax= None): 
        from rasterio import open
        from rasterio.transform import from_origin
                
        # Las coordenadas del bbox pueden venir definidas previamente tras la carga de un archivo (self.coordenada).
        # En el caso de 'energia del relieve' estas coordenadas se definen en la siguiente clase (menu) por lo que 
        # no se referencian con 'self'.
        xmin = self.xmin if xmin is None else xmin
        ymax = self.ymax if ymax is None else ymax
        
        # El 'transform' es necesario para definir la georreferenciación del raster.
        transform = from_origin(xmin, ymax, resolucion, resolucion) # (xmin, ymax, resolucion = xsize, resolucion = ysize) 
                                                                                        
        # Para guardar el array se emplea 'open' con la opción 'w' (write):
        with rasterio.open(
            ruta_salida,
            'w',
            driver='GTiff',                 # Indica el formato: GeoTiff
            height = raster.shape[0],
            width = raster.shape[1],
            count = self.bandas,
            dtype = raster.dtype,           # Indica el tipo de datos del array: en nuestro caso 'float32' (es la forma en como se definen en la 
                                            # función anterior, que se suele usar de forma conjunta con esta)
            crs = self.crs,
            transform = transform
        ) as resultado:
            resultado.write(raster,self.bandas) # Es necesario indicar de nuevo el numero de bandas para evitar 
                                                # problemas en la ejecución con archivos raster multidimensionales 
                                                # (problemas con la librería 'folium')

    #-----------------------------------------------------------------------------------------------------------------------------------
   
    # Función de visualización de los resultados raster y vectorial: devuelve un mapa interactivo con: 
    # 1) OSM como mapa base 
    # 2) Una capa de sombreado generada a partir de un MDT (entrada del usuario)
    # 3) Una capa con los polígonos que definen las zonas de escalada de nuestra zona de estudio 
    #   Fuentes: 
    #       a) 'https://www.thecrag.com/es/escalar/spain/bilbao-burgos-area'
    #       b) 'https://mapas.cantabria.es/'
    #
    # 4) Una segunda capa raster, con la variable de estudio en cuestión, o vectorial, con las unidades
    # geológicas idóneas para la escalada. 

    # Para ajustar la función a las distintas opciones del menú con las operaciones de este proyecto 
    # se han definido los siguientes argumentos: 
    # - 'ruta_lito': indica si la segunda capa es raster o vectorial.
    # - 'metodo': como se explicó anteriormente para la función 'abrir_array', indica el método de interpolación empleado en la reproyección
    #             al crs del mapa base OSM (epsg:4326). En el caso de 'energía del relieve' se define 'media' como el método de interpolación,
    #             en el resto de casos se emplea el método bilineal (por defecto). 
    # - 'estad_leyenda': estadísticas de la leyenda. Por defecto se usa el mínimo y el máximo para definir el rango de valores de la leyenda,
    #                    pero en algunos casos se han empleado los percentiles 2 y 98 en su lugar para evitar el efecto de los outliers.   
    
    def mapa(self, ruta_mdt, ruta_escalada, ruta_raster = None, titulo='Título por definir', ruta_lito= None, metodo='', estad_leyenda= ''):
    # Función: mapa
    # -------------
    # Genera un mapa interactivo combinando:
    # 0) Importación de las librerías necesarias
    # 1) Sombreado a partir de un MDT
    # 2) Raster con la variable de estudio
    # 3) Unidades geológicas 
    # 4) Zonas de escalada 
        
        #--------------------------------------------#
        # 0) Importación de las librerías necesarias #
        #--------------------------------------------#
        
        # La librería folium se emplea para generar mapas interactivos a partir de rasters en formato imagen RGB o vectorial (.shp)
        import folium                                                         
        # Dado que el formato raster que empleamos es '.tif' necesitamos convertirlo a RGB (.PNG). Para ello, primero lo abrimos como array, 
        # y luego normalizamos al rango de valores 0-255, que se emplea en una imagen RGB: 
        # 1) - Para convertir los arrays a formato PNG se emplea la libreria 'PIL' (Python Imaging Library).
        from PIL import Image                                                  
        # 2) - Para poder guardar la imagen en memoria se emplea 'io.BytesIO'.
        import io
        # 3) - La librería folium además necesita que los datos de la imagen se codifiquen en un string, 
        # para lo que se usa la librería 'base64' (se explica más adelante en detalle). 
        import base64                                                      
        # 4) La función 'LightSource' se usa para generar el sombreado, y 'Normalize' para normalizar los valores del raster a 0-1 de forma que se 
        # puedan aplicar correctamente con la función 'colormap' (también de la librería 'Matplotlib', transforma valores numéricos en colores). 
        from matplotlib.colors import LightSource, Normalize                   
        # 5) El submódulo 'matplotlib.pyplot' se usa en nuestro caso para acceder a los colormaps que se emplearán en la visualización. P.e:  
        # 'Terrain'.
        import matplotlib.pyplot as plt                                       
        # 6) El submódulo 'branca.colormap' se usa para generar la leyenda.
        import branca.colormap as cm # No confundir con el módulo 'plt.cm.terrain'                                                                                  

        
                                            #---------------------------------#
                                            # 1) Sombreado a partir de un MDT #
                                            #---------------------------------#

        # Sino se define la ruta del MDT correctamente devuelve un mensaje, p.e. si el usuario 
        # cierra la ventana para seleccionar el archivo
        if ruta_raster is None:
            print('No se ha introducido un archivo raster para generar el sombreado del mapa')
        else: # En el caso de que la ruta sea correcta, si el formato del archivo no es '.tif' 
              # imprime un mensaje informando del error:
            if not ruta_raster.endswith('.tif'):
                print('El formato del archivo raster no es válido, debe ser \'.tif\'')
        
        # Abrimos el MDT en formato array:
        mdt = self.abrir_array_raster(ruta_mdt, reproyectar=True)
        
        # Calcular sombreado
        # En primer lugar creamos el objeto de iluminación en el que se definen los grados de 
        # azimut (azdeg) y de altitud de la fuente de luz (altdeg) que se aplicarán al sombreado.
        ls = LightSource(azdeg=315, altdeg=45) # 315º de azimut y 45º de altitud

        #if mdt.ndim == 3: 
        #    mdt = mdt[0]

        # A partir del objeto creado anteriormente aplicamos la función 'hillshade' (sombreado)
        # indicando el MDT de entrada (array), la exageración vertical del sombreado (por defecto 
        # es igual a 1) y, el ancho y alto del píxel.  
        # El resultado es un array con valores float entre 0 y 1 que definen el sombreado del relieve: 
        hillshade = ls.hillshade(mdt, vert_exag=1, dx=self.resolucion, dy=self.resolucion)
        
        # Función para convertir array RGB float a base64 PNG
        def to_png_base64_rgb(array_rgb):
            # El formato de los valores del array debe ser 8bits sin signo que es el propio para RGB,
            # para lo cual usamos la función de la librería numpy 'np.uint8'. 
            # Dado que los valores del sombreado están comprendidos entre 0 y 1 sólo tenemos que multiplicarlos 
            # por 255 para tener el rango de valores de color en una imagen.
            array_255 = np.uint8(array_rgb * 255)
            # Una vez tenemos los valores en el formato correcto aplicamos la función 'Image.fromarray' de la 
            # librería PIL (Pillow), indicando el modo 'RGB', para convertir el array en imagen. 
            image = Image.fromarray(array_255, mode="RGB")
            # Finalmente guardamos la imagen generada en memoria, para lo cual tenemos que crear primero 
            # un objeto vacío en memoria con la función 'BytesIO' de la librería 'io'.
            buf = io.BytesIO()
            # Luego este contenedor se usa para guardar la imagen en formato PNG con la función 'save' de la
            # librería PIL.
            image.save(buf, format='PNG')
            # Para poder usar la imagen en memoria ('buf') con la librería 'folium' necesitamos incrustar la imagen  
            # codificada en base64 en un data URI. Un data URI es un tipo de URL que contiene los datos de la imagen,
            # de forma que puedan ser renderizados por el navegador web. 
            # Presentan la siguiente estructura  -->  data:[<tipo_mime>][;base64],<datos_codificados>
            # Para codificar los datos seguimos los siguientes pasos:
            # 1º) 'buf.getvalue()' --> devuelve los bytes de la imagen PNG
            # 2º) 'b64encode()' --> codifica los bytes anteriores en base64 (caracteres ASCII, como los de un string)
            # 3º) '.decode('utf-8') --> convierte a string UTF-8 (es formato de codificación de caracteres habitual 
            # para incluir caracteres especiales como tildes).
            # 4º) Incrustamos la imagen anterior (codificada con un string en base64 UTF-8) en un data URI, 
            # añadiéndola a: f"data:image/png;base64,{...}"
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        # El sombreado se representa en escala de grises, lo que se consigue si damos los mismos valores a cada banda (R, B y G).
        # Para ello se usa la función de numpy 'stack', con la que apilamos la lista de 3 arrays idénticos que hemos creado con:
        # '[hillshade]*3'
        hillshade_rgb = np.stack([hillshade]*3, axis=-1) # 'axis=-1' indica que el nuevo eje (dimensión) se añada al final
        # Una vez tenemos el array con los valores para las 3 bandas aplicamos la función anterior para generar una imagen PNG 
        # incrustada en un data URI.
        hillshade_url = to_png_base64_rgb(hillshade_rgb)
        
        # El mapa interactivo se genera con la librería 'folium'. 
        # Comenzamos creando el objeto 'folium.Map' y añadiendo el mapa base OSM:
        # - La localización viene dada por las coordenadas medias del bbox. 
        # - Se ha establecido un zoom inicial de 15 que coincide con la extensión de las dos zonas de estudio de nuestro proyecto. 
        # - 'control_scale = True' permite al usuario modificar el zoom. 
        m = folium.Map(location=[(self.ymin + self.ymax) / 2, (self.xmin + self.xmax) / 2], 
                       zoom_start=15, control_scale=True, tiles="OpenStreetMap")

        # Límites del bbox para la zona de estudio:
        raster_bounds = [[self.ymin, self.xmin], [self.ymax, self.xmax]]

        # A continuación, añadimos la capa de sombreado del relieve sobre el mapa anterior, indicando una opacidad de 0.6.
        # Para ello se usa la función 'ImageOverlay' que nos permite añadir una imagen como capa del mapa. 
        folium.raster_layers.ImageOverlay(
            image=hillshade_url,
            bounds= raster_bounds,
            opacity=0.6,
            name="Sombreado"
        ).add_to(m)

        
                                            #--------------------------------------#
                                            # 2) Raster con la variable de estudio #
                                            #--------------------------------------#
        
        # En el caso de que la segunda capa no sea un vectorial.
        if ruta_lito is None:
            # Abrimos el raster como array empleando la función creada para ello.
            # Como en este caso el raster se abre para añadirlo al mapa, es necesario reproyectarlo al 'epsg:4326'
            # usando el método indicado para cada caso. Como se explicó anteriormente, por defecto se emplea el 
            # método bilineal, pero en el caso de 'energía del relieve' se usa la 'media' para evitar la pérdida
            # de datos y la amplificación de los outliers. 
            array = self.abrir_array_raster(ruta_raster, reproyectar = True, metodo=metodo) 
            if array.ndim == 2: # El array debe tener 2 dimensiones, ya que 'folium' no soporta mapas en 3D.
                height, width = array.shape
            else:
                raise ValueError("El array del raster tiene una forma inesperada.")
            
            xmin_raster = self.xmin
            xmax_raster = self.xmax
            ymin_raster = self.ymin
            ymax_raster = self.ymax

            # Cambiamos los valores considerados sin datos (-9999) por nan (NotANumber), 
            # que es el formato que usa numpy.  
            array_nan = np.where(array == -9999, np.nan, array)
            # valores = array_nan[~np.isnan(array_nan)]             

            # Si el argumento 'estad_leyenda' es 'percentil2_98' se definen estos percentiles 
            # para establecer los valores mínimo y máximo que se tomarán para la leyenda y en la
            # normalización del array (a valores entre 0 y 1).
            # Para ello se usa la función numpy 'nanpercentile' (excluye los 'nan' en los cálculos).
            if estad_leyenda == 'percentil2_98':
                vmin, vmax = np.nanpercentile(array_nan,2), np.nanpercentile(array_nan,98) 
            #elif estad_leyenda == 'percentil5_95':
            #    vmin, vmax = np.nanpercentile(array_nan,5), np.nanpercentile(array_nan,95) 
            # En el resto de caso se toman los valores mínimo y máximo con las funciones numpy 'nanmin' y 'nanmax'
            else: 
                vmin, vmax = np.nanmin(array_nan), np.nanmax(array_nan)
            
            # Preconfiguramos la función 'Normalize' con los valores definidos como mínimo y máximo.
            norm = Normalize(vmin=vmin, vmax=vmax) 

            # Para transformar los números en colores se define un colormap (hemos indicado el mapa de colores 'terrain')
            cmap = plt.cm.terrain
            
            # Obtener RGBA (4 canales)
            array_norm = norm(array_nan) # Normalizamos el array.
            array_rgba = cmap(array_norm)  # Transformamos los valores del array normalizado a colores RGBA del colormap 'terrain'.
            
            # Accedemos a la cuarta banda o banda alfa (array_rgba[..., 3]) que define la transparencia para cambiar los 
            # nan por 0 (transparencia) y el resto de valores por 1 (opacidad). 
            array_rgba[..., 3] = np.where(np.isnan(array_norm), 0, 1) 

            # Función para convertir array RGBA float a base64 PNG. 
            # El funcionamiento es el mismo que el explicado anteriormente para la función 'to_png_base64_rgb',
            # pero en este caso se indica el modo 'RGBA' (al convertir el array en imagen), que nos permite  
            # añadir transparencia (banda A) para los valores nodata.
            def to_png_base64_rgba(array_rgba):
                array_255 = np.uint8(array_rgba * 255)
                image = Image.fromarray(array_255, mode="RGBA")
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            array_url = to_png_base64_rgba(array_rgba)
            
            # Definimos límites del raster que se va a añadir como capa al mapa:
            raster_bounds_raster = [[ymin_raster, xmin_raster], [ymax_raster, xmax_raster]]
            # Raster encima con opacidad 0.6.
            # Añadimos el ráster de la variable de estudio al mapa.
            folium.raster_layers.ImageOverlay(
                image=array_url,
                bounds=raster_bounds_raster,
                opacity=0.6,
                name=titulo
            ).add_to(m)

            # Finalmente, creamos la leyenda que se usará en la representación anterior,
            # para ello usamos el módulo 'colormap' de la librería 'branca'.
            import branca.colormap as cm
            from matplotlib import colors
            
            # Creamos un array de 256 valores que van de 0 a 1 con la función 'linspace' de 
            # numpy, y usamos la función 'terrain' para transformar estos valores en colores 
            # del colormap de 'terrain'. 
            rgba_colores = plt.cm.terrain(np.linspace(0, 1, 256)) 
            
            # Convertimos cada color RGBA a código hexadecimal para que se puedan usar con 
            # 'folium' y 'branca'. 
            hex_colores = [colors.to_hex(c) for c in rgba_colores]

            # Creamos el mapa de colores para la leyenda:
            colormap = cm.LinearColormap(
                colors=hex_colores,
                vmin=vmin,
                vmax=vmax,
                caption=titulo, # Título que se mostrará en la leyenda.
            )
            # Definimos el ancho del mapa en 500 píxeles.
            colormap.width = 500  
            # Añadimos la leyenda creada al mapa.
            colormap.add_to(m)

        
                                                #------------------------#
                                                # 3) Unidades geológicas #
                                                #------------------------#

        # En el caso de que se cargue el vectorial de litología:
        if ruta_lito is not None:
            # Para abrir un vectorial se emplea la función 'read_file' de la librería 'geopandas'
            unidades_lito = gpd.read_file(ruta_lito)
            # En el caso de que el vectorial no tenga asignado crs asignamos el 'epsg:25830' con 
            # la función 'set_crs' de la librería 'geopandas'.
            if unidades_lito.crs is None:
                unidades_lito = unidades_lito.set_crs(self.crs)
                # Luego reproyectamos el vectorial al crs de OSM ('epsg:4326') con la función 
                # 'to_crs' de la librería 'geopandas'.
            unidades_lito = unidades_lito.to_crs("EPSG:4326")
            
            # Colores únicos para cada valor de 'Descripcio'
            import matplotlib.colors as mcolors # Este módulo se emplea para trabajar con mapas de colores (colormaps) 
            # Tomamos los valores únicos del campo 'Descripcio' del vectorial de 'unidades litologicas'.
            valores_unicos = unidades_lito["Descripcio"].unique() 
            # A continuación aplicamos la paleta de colores 'tab20' que contiene 20 colores predefinidos. Con la función
            # 'get_cmap' y el atributo 'len(valores_unicos)' indicamos que el número de colores a usar de la paleta sea 
            # el mismo que de valores únicos encontrados en 'unidades litologicas'. 
            cmap = plt.get_cmap("tab20", len(valores_unicos))

            # Finalemente se genera un diccionario de colores asignando a cada valor único un color en formato hexadecimal, 
            # con la función 'to_hex', de forma que la estructura del diccionario sea la siguiente: {val: valor hexadecimal}.
            colores = {val: mcolors.to_hex(cmap(i)) for i, val in enumerate(valores_unicos)} # La función'enumerate' devuelve tanto el 
                                                                                            # índice como el valor de cada elemento de un array.
            
            folium.GeoJson(
                unidades_lito,
                name="Unidades Geológicas",
                # El estilo el 'folium' lo definimos a través de una función (por simplificar se emplea una función lambda) 
                # que genera un diccionario con las propiedades de estilo que se aplicarán en la representación del vectorial.
                style_function=lambda unidad: {
                    # 'color' define el color de las líneas del polígono. 
                    "color": colores.get(unidad["properties"]["Descripcio"], "white"), # A partir del diccionario de colores anterior asignamos un  
                                                                                       # color a cada valor único del campo 'Descripcio',  y en el 
                                                                                       # caso, de que no encuentre un color usa blanco por defecto. 
                    "weight": 0.1, # Grosor
                    "fillOpacity": 0.6, # Opacidad
                    "fillColor": colores.get(unidad["properties"]["Descripcio"], "white"), # Color de relleno del polígono.
                },
                # Con el módulo 'GeoJsonTooltip' de 'folium' podemos generar una ventana emergente que nos muestre información 
                # de la tabla de atributos para un elemento cuando pasamos el cursor sobre este. 
                tooltip=folium.GeoJsonTooltip( 
                    fields=["Sistema", "Descripcio"], # Tomamos los campos 'Sistema' y 'Descripcio' de 'unidades_lito'.
                    aliases=["Sistema: ", "Unidad Geológica: "] # Asignamos un alias a los nombres de los campos anteriores. 
                )
            ).add_to(m) # Añadimos la capa de 'unidades litologicas' al mapa.

        
                                                #------------------------#
                                                # 4) Zonas de escalada   #
                                                #------------------------#

        # Cargamos el vectorial de zonas de escalada:
        zonas_escalada = gpd.read_file(ruta_escalada)
        # Si no tiene crs asignado, asignamos 'epsg:25830' como en el caso anterior.
        if zonas_escalada.crs is None:
            zonas_escalada = zonas_escalada.set_crs(self.crs)
        zonas_escalada = zonas_escalada.to_crs("EPSG:4326") # Reproyectamos al crs de OSM.
        
        folium.GeoJson(
            zonas_escalada,
            name="Zonas de Escalada",
            # En este caso definimos un sólo estilo para todos los polígonos del vectorial:
            style_function=lambda feat: {"color": "red", "weight": 2, "fillOpacity": 0}, # Bordes rojos, grosor 2 y sin relleno.
            # Definimos una ventana emergente para que muestre información de cada zona de escalada, en este caso mostramos
            # los campos 'Nombre' y 'No_Rutas' con los alias 'Zonas de escalada' y 'Número de rutas', respectivamente.
            tooltip=folium.GeoJsonTooltip(fields=["Nombre", "No_Rutas"], aliases=["Zona de escalada: ", "Número de rutas"]) 
        ).add_to(m) # Añadimos la capa de 'zonas de escalada' al mapa.
        
        # Finalmente añadimos una herramienta para controlar interactivamente las capas que se muestran en el mapa.
        folium.LayerControl(collapsed=False).add_to(m) # 'Collapsed = False' --> El panel está desplegado y no se puede ocultar.
        
        return m # Tenemos que explicitar el return del mapa para poder reutilizarlo en otras funciones de la clase 'menu()'.


    #-----------------------------------------------------------------------------------------------------------------------------------

    # Cálculo del MDT a partir de un archivo en formato LAS/LAZ 
    def ejec_las2mdt_r(self,script_ruta):
        # La librería 'subprocess' permite ejecutar comandos como si se hiciera desde consola. 
        import subprocess

        # Comprobamos si existe la ruta al script de R:
        if not os.path.exists(script_ruta):
            raise FileNotFoundError(f"Script R no encontrado: {script_ruta}")
        
        # Indicamos la ruta al ejecutable de R que ejecutará el script:
        rscript_path = r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"
        # En el caso de que la ruta de 'Rscript.exe' sea distinta en la máquina 
        # del usuario devuelve el error tipo 'archivo no encontrado'
        if not os.path.exists(rscript_path):
            raise FileNotFoundError(f"Rscript no encontrado: {rscript_path}")
        
        # Probamos la ejecución del script para controlar los errores sin para el resto de la ejecución.
        try:
            # 'subprocess.run' permite ejecutar comandos desde consola (script en este caso con 
            # el ejcutable 'Rscript.exe'). 
            resultado = subprocess.run(
                [rscript_path, script_ruta],
                capture_output=True, # Guarda la salida en 'resultado' (stdout) y el error en 'e' (stderr)
                text=True,           # Convierte la salida en texto (por defecto son bytes)
                check=True           # Lanza error si se produce un error en la ejecución del script dentro de R.
            )
            print("Salida R:\n", resultado.stdout)
        except subprocess.CalledProcessError as e:
            print("Error en la ejecución de R:")
            print(e.stderr)
        
    #-----------------------------------------------------------------------------------------------------------------------------------
                                                
                                                ############### OPCION 1 DEL MENÚ ############### 
    
    # Para calcular la pendiente creamos una función en la que indicamos como argumento, además de las rutas de entrada y salida, 
    # las unidades de la pendiente.
    def pendiente(self, ruta_mdt, ruta_salida, formato_pendiente='Degree'):    
        # Para abrir u procesar los datos del MDT, en este caso, usaremos la librería GDAL.
        from osgeo.gdal import DEMProcessingOptions

        mdt = gdal.Open(ruta_mdt)
        if not mdt:
            print('No se ha podido cargar el MDT')
        # Configuramos las opciones para calcular las pendientes (en nuestro caso sólo el formato de pendiente).
        options  = DEMProcessingOptions(slopeFormat = formato_pendiente)
        # Para calcular pendientes usamos la función 'DEMProcessing' e indicamos 'slope' como argumento, 
        # ya que se emplea en otros cálculos también como el sombreado o la orientación. 
        pendientes = gdal.DEMProcessing(ruta_salida, ruta_mdt, 'slope', options=options)

    
                                                ############### OPCION 2 DEL MENÚ ############### 
   
    # Partiendo del archivo generado anteriormente para pendientes, en la siguiente función filtraremos para 
    # quedarnos sólo con pendientes de más de 60º. Para ello tenemos que transformar el resultado anterior en array 
    # con la función 'abrir_array' creada anteriormente.  
    def pendiente_selec(self, ruta_pendiente, ruta_salida, nombre_pendientes ='pendientes', nombre_salida = '_60.tif', 
                        pendiente_valor=60): 
        
        pend_array = self.abrir_array_raster (ruta_pendiente)
        # La función 'where' de 'numpy' nos permite modificar los valores de array que cumplan una determinada condición, 
        # en nuestro caso, que las pendientes sean superiores a 60º. Si se cumple la condición se mantiene el valor de 
        # pendiente original, y en caso contrario hemos asignado el 'nan' de 'numpy' con constante 'np.nan'.
        pendientes60 = np.where(pend_array > pendiente_valor, pend_array,np.nan)
        pendientes60 = pendientes60.squeeze() # Elimina las dimensiones de valor/tamaño 1, quedarnos con una array 2D. 
        # Dado que en SIG se suele usar -999 como 'ndata' debemos de sustituir los 'nan' por '-9999'. Para ello 
        # usamos la función de 'numpy' 'nan_to_num'. 
        pendientes60_out = np.nan_to_num(pendientes60, nan=-9999).astype('float32') # Formato decimal de 32bits.
            
        xmin = self.xmin
        ymax = self.ymax
        # Guardamos el resultado con la función definida al inicio:
        self.guardar_array(pendientes60_out, ruta_pendiente.replace('.tif', nombre_salida), 
                           xmin= xmin, ymax= ymax, resolucion = self.resolucion)
        # Es necesario usar 'return' para usar el resultado directamente en otra función.
        return pendientes60
        
#-----------------------------------------------------------------------------------------------------------------------------------
    
                                                ############### OPCION 3 DEL MENÚ ############### 
  
    # La energía del relieve se calcula como la diferencia entre el MDT original y la suma de la media y la desviación típica 
    # (calculadas para un determinado tamaño de ventana, en nuestro caso 25m): 
    #                                                    ER = MDT - (M̅D̅T̅ + σ_mdt)
    
    def energia_relieve(self, ruta_mdt, ruta_salida, ventana=25): 
        # Para calcular estadísticas focales se emplea el módulo 'generic_filter' de la librería 'scipy'.
        from scipy.ndimage import generic_filter
        
        mdt = gdal.Open(ruta_mdt)
        if not mdt:
            print('No se ha podido cargar el MDT')
        # En este caso abriremos, el MDT con GDAL y lo transformamos en array con 'ReadAsArray' para agilizar
        # la ejecución (no necesitamos una configuración como la definida en la función 'abrir_array').
        mdt_array = mdt.ReadAsArray()
        
        # Calculamos la media y la desviación típica focales con la función 'generic_filter' indicando como 
        # segundo argumento la función estadística que calcularemos, es decir, 'nanmean' y 'nan_std', 
        # respectivamente (que excluyen los 'nan' en los cálculos).
        mdt_focal_media = generic_filter (mdt_array, np.nanmean, size = ventana, mode='nearest') # 'mode' indica el modo que se usará para tratar los  
        mdt_focal_std = generic_filter (mdt_array, np.nanstd, size = ventana, mode='nearest')    # bordes del array. 'nearest' --> 'vecino más cercano'
        mdt_mean_std = mdt_focal_media + mdt_focal_std
        dif_mdt_focal = mdt_array - mdt_mean_std
        
        # Del resultado anteior sólo nos interesan los valores positivos, por lo que usaremos 'np.where' 
        # para sustituir los valores que no cumplen esa condición por 'nan'. 
        en_relieve = np.where((dif_mdt_focal>0), dif_mdt_focal, np.nan)

        # La función 'GetGeoTransform' devuelve una tupla de valores ligada la transformación geoespacial, 
        # de forma que el primer y el cuarto valor representan las coordenadas 'xmin' y 'ymax'.
        geotransform = mdt.GetGeoTransform()
        xmin = geotransform[0] # xmin
        ymax = geotransform[3] # ymax

        # Finalmente, guardamos el array con la función creada al inicio de la clase:
        self.guardar_array(en_relieve, 
                           ruta_salida,
                           resolucion = 2, xmin = xmin, ymax = ymax
                          )
        
#-----------------------------------------------------------------------------------------------------------------------------------
    
                                                ############### OPCION 4 DEL MENÚ ############### 

    # En la siguiente función se definen los parámetros que se emplearán para el cálculo de las variables derivadas del MDT. 
    # Las zonas de estudio tienen un relieve pronunciado, por lo que la configuración de estos parámetros es aplicable a 
    # zonas de relieve similar. La variable a calcular se indica en el argumento 'analisis'. 
    def herramientas_rvt(self, analisis, ruta_mdt, ruta_salida): 
        # El módulo 'vis' de la librería 'Relief Visualization Tools' contiene las funciones para generar las variables 
        # derivadas del MDT que usaremos en este proyecto.
        import rvt.vis
        # 'rvt.default': se usa con el submódulo 'save_raster' para guardar el resultado y establecer la configuración 
        # de parámetros/argumentos, p.e el valor 'nodata'. 
        import rvt.default
        
        mdt_array = self.abrir_array_raster(ruta_mdt, False)

        # La función 'lower' cambia un string a minúsculas para homogeneizar el formato y evitar posible errores 
        # al introducir el nombre para el argumento 'analisis'.
        if analisis.lower() == 'ld': # Local Dominance: inclinación media con la que un observador mira hacia abajo la superficie 
                                     # dentro de los radios mínimo y máximo definidos a continuación: 
            min_rad = 2              # radio mínimo 
            max_rad = 5              # radio máximo 
            rad_inc = 1              # incremento del radio 
            angular_res = 15         # resolución angular (15º)
            observer_height = 1.7    # altura del observador (1.70m)
            
            local_dom_arr = rvt.vis.local_dominance(dem=mdt_array,
                                                min_rad = min_rad,
                                                max_rad = max_rad,
                                                rad_inc = rad_inc,
                                                angular_res = angular_res,
                                                observer_height = observer_height,
                                                ve_factor=1,
                                                no_data = -9999)
            # Creamos un array vacío de las mismas dimensiones que 'local_dom_arr', de esta forma evitamos 
            # errores de 'shape mismatch' ya que en este caso encontramos valores 0 o negativos además 
            # de no_data (-9999) y al filtrar estos valores (previo al calculo del log) se modifican
            # las dimensiones del array. 
            # 1º creamos un array lleno de 'nans' con la función 'full_like' de 'numpy'.
            local_dom_arr_ln = np.full_like(local_dom_arr, np.nan, dtype=float)
            # 2º tomamos como máscara sólo los valores positivos.
            mask = local_dom_arr > 0
            # 3º, partiendo de la máscara anterior, añadimos al array vacío el logaritmo del resultado 
            # para 'local-dominance'.
            local_dom_arr_ln[mask] = np.log(local_dom_arr[mask])            
            # Finalmente, guardamos el resultado en memoria.
            self.guardar_array(local_dom_arr_ln, ruta_salida)
        

    
                    #----------------------############### OPCION 5 DEL MENÚ ###############----------------------#
    
        elif analisis.lower() == 'svf': # Sky-View Factor (Zakšek et al. 2011): representa la proporción de cielo visible desde 
                                        # el punto de vista del observador, de forma que valores próximos a 1 indican zonas de  
                                        # relieve abrupto como un cañón; mientras que valores cercanos a 0 indican que está 
                                        # despejado y no hay obstáculos por el relieve.  
            svf_n_dir = 16  # número de direcciones que se emplean en el cálculo .
            svf_r_max = 10  # radio de búsqueda máximo en píxeles (tamaño de la ventana) --> En este caso no se usa, se emplea en el cálculo de 
                            # 'Openness', pero es necesario definir este parámetros para evitar errores en la ejecución del módulo.
            svf_noise = 0  # nivel de eliminación de ruido (0-no eliminar, 1-bajo, 2-medio, 3-alto) --> Indicamos 0 para evitar el efecto de los 
                           # filtros de suavizado que se aplican para eliminar el ruido, ya que estos detalles podrían representar elementos del 
                           # relieve como 'bloques' que sean idóneos para la escalada. 
            
            asvf_level = 1  # nivel de anisotropía (1-bajo, 2-alto) --> En la fórmula, Zakšek et al. 2012, está asociado a 'c'.
            asvf_dir = 315  # dirección de la anisotropía, es el azimut de mayor peso para calcular la anisotropía (en la
                            # fórmula, Zakšek et al. 2012, viene dado por 'lambda max').
            dict_svf = rvt.vis.sky_view_factor(dem=mdt_array, 
                                               resolution=self.resolucion, 
                                               compute_svf=False, # SVF isotrópico (no se calcula)
                                               compute_asvf=True, # SVF anisotrópico 
                                               compute_opns=False, # Openness (se calcula posteriormente con otros parámetros)
                                               svf_n_dir=svf_n_dir, 
                                               svf_r_max=svf_r_max,  
                                               svf_noise=svf_noise,
                                               asvf_level=asvf_level, 
                                               asvf_dir=asvf_dir,
                                               no_data=-9999)    
            svf_arr = dict_svf["asvf"]  # El resultado es un diccionario del que extraemos la variable de interés. El resto 
                                        # de opciones dada la configuración anterior darían 'None' si intentamos extraerlas.
            self.guardar_array(svf_arr, ruta_salida) # Guardamos el resultado en la ruta indicada 
        

    
                    #----------------------############### OPCION 6 DEL MENÚ ###############----------------------#
    
        elif analisis.lower() == 'po':  
            # El cálculo de 'Openness' es similar al de SVF, pero en este caso se define un radio de búsqueda en el cálculo del 
            # angulo de elevación horizontal. 
            svf_n_dir = 16  # número de direcciones
            svf_r_max = 10  # radio de búsqueda máximo en píxeles o tamaño de la ventana empleado en el cálculo de Openness.
            svf_noise = 0  # nivel de eliminación de ruido
            
            asvf_level = 1  # nivel de anisotropía 
            asvf_dir = 315  # dirección de la anisotropía 
            dict_svf = rvt.vis.sky_view_factor(dem=mdt_array, 
                                               resolution=self.resolucion, 
                                               compute_svf=False, 
                                               compute_asvf=False, 
                                               compute_opns=True,
                                               svf_n_dir=svf_n_dir, 
                                               svf_r_max=svf_r_max, # radio máximo definido para el cálculo de 'Openness'
                                               svf_noise=svf_noise,
                                               asvf_level=asvf_level, 
                                               asvf_dir=asvf_dir,
                                               no_data=-9999)    
            svf_arr = dict_svf["opns"]  # Extraemos el resultado del diccionario.
            
            self.guardar_array(svf_arr, ruta_salida)

            
    
                    #----------------------############### OPCION 7 DEL MENÚ ###############----------------------#
    
        elif analisis.lower() == 'no':  
            svf_n_dir = 16  # número de direcciones
            svf_r_max = 10  # radio de búsqueda máximo en píxeles o tamaño de la ventana empleado en el cálculo de Openness.
            svf_noise = 0  # nivel de eliminación de ruido
            
            asvf_level = 1  # nivel de anisotropía
            asvf_dir = 315  # dirección de la anisotropía 
            
            # PARA CALCULAR 'NEGATIVE OPENNESS' BASTA CON HACERLO A PARTIR DEL MDT CON SIGNO CAMBIADO
            dict_svf = rvt.vis.sky_view_factor(dem=mdt_array*-1, 
                                   resolution=self.resolucion, 
                                   compute_svf=False, 
                                   compute_asvf=False, 
                                   compute_opns=True,
                                   svf_n_dir=svf_n_dir, 
                                   svf_r_max=svf_r_max,  # radio máximo definido para el cálculo de 'Openness'
                                   svf_noise=svf_noise,
                                   asvf_level=asvf_level, 
                                   asvf_dir=asvf_dir,
                                   no_data=-9999)
            neg_opns_arr = dict_svf["opns"]  
            
            self.guardar_array(neg_opns_arr, ruta_salida)

        
    
                    #----------------------############### OPCION 8 DEL MENÚ ###############----------------------#
        # Para calcular los MSRMs se siguen los siguientes pasos:
        # 1) Se aplican filtros low pass con diferentes tamaños de kernel sobre el MDT original. Partimos de 'feature_min' 
        #    como tamaño de kernel y vamos incrementando el tamaño (scaling factor) en cada iteración hasta alcanzar el 
        #    tamaño máximo ('feature_max')
        # 2) Se substraen los resultados entre cada tamaño.
        # 3) Se suman las diferencias anteriores.
        # 4) Se divide entre el número de modelos del relieve.
        elif analisis.lower() == 'msrm':   
            feature_min = 10  
            feature_max = 50  
            scaling_factor = 3  
            # En este caso ejecutamos el submódulo específico para el cálculo del 'MultiScale Relief Model' ('msrm')
            msrm_arr = rvt.vis.msrm(dem=mdt_array, 
                                    resolution=self.resolucion, 
                                    feature_min=feature_min, 
                                    feature_max=feature_max, 
                                    scaling_factor=scaling_factor, 
                                    ve_factor=1, # "Exageración vertical" (Vertical Exaggeration)
                                    no_data= -9999)
        
            self.guardar_array(msrm_arr, ruta_salida)
            

    
                    #----------------------############### OPCION 9 DEL MENÚ ###############----------------------#
    
        elif analisis.lower() == 'mstp': 

            # Se definen tres escalas, aunque sólo emplearemos la escala local, ya que es la que nos muestra más detalles 
            # sobre patrones en la posición topográfica. 
            # Se calcula la desviación respecto a la media normalizada para la desviación típica (para cada píxel) empleando 
            # distintos tamaños de ventana, lo que genera varias capas (una por tamaño de ventana). 
            # Finalmente, se promedia el resultado de las distintas capas (dentro de una escala determinada). 
            # El tamaño de ventana inicialmente es 3 píxeles, y aumenta 2 píxeles en cada iteración hasta alcanzar un tamaño máximo 
            # de 21 píxeles (escala local).
            local_scale=(3, 21, 2)  # mínimo, máximo, incremento
            meso_scale=(23, 203, 18)  # mínimo, máximo, incremento
            broad_scale=(223, 2023, 180)  # mínimo, máximo, incremento
            lightness=1.2  # Se suele emplear un valor entre 0.8 y 1.6 que es el rango que genera mejores visualizaciones.
            mstp_arr = rvt.vis.mstp(dem=mdt_array, 
                                    local_scale=local_scale, 
                                    meso_scale=meso_scale,
                                    broad_scale=broad_scale, 
                                    lightness=lightness, 
                                    ve_factor=1, # "Exageración vertical" (Vertical Exaggeration)
                                    no_data=-9999)
            
            # La tercera banda es la que representa el 'mstp' a menor escala (local):   
            mstp_arr = mstp_arr[2]  
            
            # Dado que en este caso es necesario tomar información espacial de referencia no podemos aplicar 
            # la función 'guardar_array' definida al inicio de la clase. Por ello se usa el submódulo 
            # 'save_raster' del módulo 'rvt.default' 
            rvt.default.save_raster(
                src_raster_path=ruta_mdt,     # Es necesario el MDT original para tomar la información geoespacial de este y aplicarla 
                                              # en el resultado anterior, que al ser un array no contiene información geoespacial.
                out_raster_path=ruta_salida,
                out_raster_arr=mstp_arr,        
                no_data=np.nan,
                e_type=6              # 'e_type' indica el tipo de datos. '6' se corresponde con 'float32'
                )

                    #-----------------------------------------------------------------------------------------------#
        # En el caso de que no se elijan ninguna de las opciones anteriores muestra un mensaje (esto sólo es posible si el usuario
        # modifica el código original, ya que en las funciones de la clase 'menu' este argumento ya viene predefinido).
        else:
            print('Ha modificado el código original, revise el argumento \'analisis\' de las funciones RVT en ambas clases:\n',
                  '- HSMD: Hillshade from multiple directions\n',
                  '- LD: Local Dominance\n',
                  '- SVF: Anisotropic sky-view factor\n',
                  '- PO: Positive Openness\n',
                  '- NO: Negative Openness\n',
                  '- MSRM: Multiscale Relief Model\n',
                  '- MSTP: Multiscale Topographic Position\n del ')
        
        
#-----------------------------------------------------------------------------------------------------------------------------------
    
                                                ############### OPCION 10 DEL MENÚ ############### 
    
    def rugosidad(self, ruta_mdt, ruta_salida):
        from osgeo.gdal import DEMProcessingOptions
        # La rugosidad viene dada por la diferencia entre la altura máxima y mínima dentro de una 
        # ventana de un determinado tamaño (por defecto de 3x3 píxeles) (Wilson et al., 2007). 
        # Por ello no es necesario definir más parámetros como en casos anteriores. 
        rugosidad = gdal.DEMProcessing(
                                    destName=ruta_salida,
                                    srcDS=ruta_mdt,
                                    processing='roughness' # Rugosidad
                                    )
        
    # Realizaremos el cálculo del logaritmo en una función por separado para obtener los dos 
    # resultados (original y con logaritmo) en disco y simplificar el código (de lo contrario hay 
    # que guardar el resultado en memoria).
    
    def rugosidad_logn (self, ruta_mdt, ruta_salida): 
        from rasterio import open        

        self.rugosidad(ruta_mdt, ruta_salida)

        # Leer el resultado de rugosidad
        with rasterio.open(ruta_salida) as src:
            rugosidad_array = src.read(1)
            nodata = src.nodata  # normalmente -9999
            bounds = src.bounds
            self.xmin = bounds.left
            self.ymax = bounds.top    
        
        # Reemplazamos -9999 por np.nan
        rugosidad_array = np.where(rugosidad_array == nodata, np.nan, rugosidad_array)
        
        # Creamos un array vacío de las mismas dimensiones que 'rugosidad_array', de esta forma evitamos 
        # errores de 'shape mismatch' ya que en este caso encontramos valores 0 o negativos además 
        # de no_data (-9999) y al filtrar estos valores (previo al calculo del log) se modifican
        # las dimensiones del array. 
        # 1º creamos un array lleno de 'nans' con la función 'full_like' de 'numpy'.
        rugosidad_ln = np.full_like(rugosidad_array, np.nan, dtype=float)  # inicializa con nan
        # 2º tomamos como máscara sólo los valores positivos.
        mask = rugosidad_array > 0
        # 3º, partiendo de la máscara anterior, añadimos al array vacío el logaritmo del resultado 
        # para 'rugosidad_array'.
        rugosidad_ln[mask] = np.log(rugosidad_array[mask])      

        # Guardamos el resultado final añadiendo '_ln' al nombre del resultado original (sin logaritmo). 
        self.guardar_array(rugosidad_ln, ruta_salida.replace('.tif','_ln.tif'))
        
            
#-----------------------------------------------------------------------------------------------------------------------------------
    
                                                ############### OPCION 11 DEL MENÚ ############### 
    
    # En la siguiente función seleccionaremos las unidades litológicas apropiadas para la escalada dentro de la tabla de atributos 
    # de la capa vectorial de litología. 
    def procesado_vector(self, ruta_lito, campo, ruta_salida):  
        
        # Cargamos la capa vectorial
        shp_lito = gpd.read_file(ruta_lito)

        # Antes de proceder al filtrado debemos de disolver la capa por el campo que contiene las unidades litológicas (se define 
        # como argumento en la función para indicarlo posteriormente en la función correspondiente de la clase 'menu').
        # Para disolver la capa empleamos el módulo 'dissolve' de la librería 'geopandas' (importada al inicio del script).
        shp_disuelto = shp_lito.dissolve(by=campo)
        print('Capa disuelta')

        # Para determinar que elementos de la capa de litología son apropiados para la escalada se ha empleado el campo de descripción.
        # Las descripciones seleccionadas son representativas de las zonas de estudio (Cantabria) POR LO QUE EL USUARIO DEBERÍA AJUSTARLAS
        # EN EL CASO DE QUE LA LITOLOGÍA SEA DISTINTA. 
        # Para más información sobre los criterios específicos empleados en la selección de las unidades litológicas consultar la memoria 
        # del proyecto.
        descripciones_seleccionadas = [
                'Brechas calcáreas, calizas boundstone microbiales, calizas grainstone-packstone oolíticas, bioclásticas, calizas packstone y calizas bafflestone',
                'Brechas calcáreas, calizas boundstone  microbiales, calizas grainstone-packstone oolíticas, bioclásticas, calizas wackestone y bafflestone, Colores gris claro, blanco o rojizo',
                'Espiculitas, calizas mudstone finamente laminadas, fétidas, calizas grainstone-packstone y calizas rudstone (facies de talud y plataforma)',
                'Espiculitas, calizas mudstone negras, finamente laminadas, fétidas, calizas grises laminadas, calizas bioclásticas y brechas calcáreas',
                'Calizas negras finamente laminadas, fétidas',
                'Lutitas, espiculitas, calizas mudstone finamente laminadas, Calizas grainstone-packstone, calizas rudstone. Facies de talud y cuenca',
                'Espiculitas, calizas mudstone finamente laminadas, calizas grainstone-packstone, calizas rudstone (facies de base de talud y cuenca)',
                'Brechas calcáreas, calizas boundstone  microbiales, calizas grainstone-packstone oolíticas, bioclásticas, calizas wackestone y bafflestone, Colores gris claro, blanco o rojizo',
                'Bloques, cantos angulosos, arenas', 
                'Caliza margosa, cuarzo-arenita', 
                'Caliza masiva con rudistas', 
                'Caliza masiva gris con rudistas, orbitolina',
                'Calizas',
                'Calizas con rudistas en capas discontinuas',
                'Calizas con rudistas, orbitolinas e intercalaciones de arenas',
                'Calizas masivas con rudistas',
                'Margas oscuras con intercalaciones arenosas',
                'Margocaliza, caliza margosa, arenisca, marga oscura laminada'
            ]
        
        print('Capa con descripciones seleccionadas creada correctamente')
        # Una vez tenemos las descripciones apropiadas para nuestra zona de estudio, nos quedamos sólo con 
        # aquellos elementos de la capa que coincidan con alguna de las descripciones. Para ello se indexa 
        # dentro de la capa empleando el módulo 'isin' de 'geopandas'. 
        poligonos_selec = shp_disuelto[shp_disuelto['Descripcio'].isin(descripciones_seleccionadas)]
        # Para guardar el resultado final empleamos el módulo 'to_file' de 'geopandas'.
        poligonos_selec.to_file(ruta_salida)
        return ruta_salida

            
#----------------------------------------------------------------------------------------------------------------------------------
    
    #------------ OPCION 12 DEL MENÚ (Parte 1: rasterización de las unidades litológicas apropiadas para la escalada) ------------#
    
    
    # Como sólo se usa en el Análisis Multicriterio no es necesario introducir el argumento 'analisis_escalada' .
    # Antes de proceder con el cálculo de las zonas potenciales de escalda debemos de tener en cuenta la 
    # litología en el mismo, por ello tenemos que rasterizar el resultado de la función anterior.
    def rasterizar_vectorial(self, ruta_vect, ruta_salida, valor=1.00): # 1 es el valor que asignaremos a las zonas 
                                                                        # apropiadas litologicamente para la escalada.
        # Como indica su nombre el submódulo 'rasterize' de la librería 'rasterio' se emplea para convertir una capa vectorial en raster.
        from rasterio.features import rasterize
        # Como en casos anteriores, el submódulo 'from_origein' se usa para crear el 'transform', que contiene la información 
        # geoespacial que se aplicará en el nuevo raster.
        from rasterio.transform import from_origin
        
        nombre = 'lito_escalada'
            
        # Leemos el vectorial
        vector = gpd.read_file(ruta_vect)
        if vector.empty:
            raise ValueError("El archivo vectorial no contiene geometrías. Verifica que el SHP tenga datos.")
        
        # Calculamos las dimensiones del raster en píxeles
        # Las coordenadas del bbox se inicializan al llamar previamente a la función 'abrir_arrary_raster'
        self.ancho = int((self.xmax - self.xmin) / self.resolucion) 
        self.alto = int((self.ymax - self.ymin) / self.resolucion)
        
        # Creamos transform (origen en la esquina superior izquierda)
        transform = from_origin(self.xmin, self.ymax, self.resolucion, self.resolucion)
        
        # En último lugar, creamos una lista de (geom, value) para rasterize de cada elemento (1:coincidente con la descripción, 
        # 0: no apropiado para escalada). Los valores inicialmente están vacíos. Esta lista contiene todos los elementos de la 
        # capa y un espacio para indicar el valor en la rasterización.
        vectores_valor = [(geom, valor) for geom in vector.geometry]
        print('Capa vectorial con valor 1 para las zonas seleccionadas creada correctamente')
        # Rasterizar
        raster = rasterize(
            vectores_valor,
            out_shape=(self.alto, self.ancho), 
            transform=transform,
            fill=0, # Valor para el fondo (no coincide con las descripciones anteriores).
            default_value=valor, # Es el valor que se dará a los elementos coincidentes (1).
            dtype='uint8' # Entero sin signo 8 bits.  
        )

        self.guardar_array(raster,os.path.normpath(ruta_salida))
        print('Archivo rasterizado con valor 1 para las zonas seleccionadas correctamente')
        return os.path.normpath(ruta_salida) # Posteriormente usaremos el archivo en disco, por lo que basta sólo con devolver la ruta. 
                                                
                    
    #---------------------- OPCION 12 DEL MENÚ (Parte 2: Cálculos de las zonas potenciales de escalada) ----------------------#
    # Para calcular el potencial de escalda daremos distintos pesos en función de como afecten al cálculo. 
    # Finalmente, descartaremos las zonas que no tengan la litología apropiada. 
    def analisis_escalada (self, ruta_vect, ruta_pend, ruta_er, 
                                       ruta_ld, ruta_svf, ruta_po, ruta_msrm, ruta_mstp, 
                                       ruta_rug, ruta_lito_selec_raster, ruta_salida):
        # El submódulo 'zonal_stats' se emplea para clular las estadísticas zonales a partir de cada una de las zonas de 
        # escalada del vectorial.
        from rasterstats import zonal_stats
        # 'box' nos permite delimitar el bbox de la zona de estudio (para recortar las zonas de escalada que sobrepasen 
        # la zona de estudio).
        from shapely.geometry import box
        # El submódulo 'unary_union' se emplea para unir todas las geometrías en una sola.
        from shapely.ops import unary_union
        # Inicializamos las variables que usaremos posteriormente.
        suma_arrays = None
        raster_validos = 0
        zonas_escalada = gpd.read_file(ruta_vect)
        # Creamos una lista con las rutas de tods los derivados del MDT, que se han calculado previamente, para poder 
        # procesarlos en un solo bucle 'for'.
        rutas_raster = [ruta_pend, ruta_er, 
                        ruta_ld, ruta_svf,
                         ruta_po, ruta_msrm, ruta_mstp, 
                         ruta_rug, ruta_lito_selec_raster] 
        # Creamos un polígono con el bbox de la zona de estudio para cortar las zonas de escalada que la excedan.        
        rect = box(self.xmin, self.ymin, self.xmax, self.ymax)
        # Antes de comenzar nos quedaremos con las zonas de escalada situadas dentro de la zona de estudio. 
        # Para ello en primer lugar filtramos las zonas que se encuenrtan dentro on intersectan empleando 
        # el sbmódulo 'intersects' de 'geopandas'.
        zonesc_clip = zonas_escalada[zonas_escalada.geometry.intersects(rect)].copy() 
        # Luego cortamos las zonas resultantes pra que se ajustan al bbox, para ello se usa el subomódulo 
        # 'intersection' de 'geopandas'.
        zonesc_clip = zonesc_clip.set_geometry(zonesc_clip.geometry.intersection(rect)) 
        # Con 'unary_union' convertimos la capa con las zonas de escalada en una sola geomtría (en este caso multigeometría).
        geometria_completa = unary_union(zonesc_clip.geometry)
        
        for ruta_raster in rutas_raster:
            # Calculamos el mínimo y máximo dentro de las zonas de escalada para cada variable.
            estad_zona = zonal_stats(geometria_completa, ruta_raster, 
                                     stats = ['min','max'],
                                     nodata = None
                                     )[0] # 'zonal_stats' devuelve por defecto una lista de diccionarios por lo que tenmos 
                                          # que extraer el primer elemento.
            # Para obtener el valor mínimo y máximo del diccionario anterior usamos 'get'.
            minimo = estad_zona.get('min')
            maximo = estad_zona.get('max')
            # Usamos la función creada inicialmente para abrir los distintos raster.
            arr = self.abrir_array_raster(ruta_raster)
            # Definimos el mismo tamaño (ancho y alto) para todos los arrays, de esta forma evitamos errores en el cálculo. 
            # Estas variables son declaradas en módulos anteriores de esta clase.  
            array = arr[:self.alto,:self.ancho]
            # Filtramos los valores que se encuentran dentro del rango definido por el valor máximo y mínimo de las zonas 
            # de escalada para una variable determinada. En el resto de casos asignamos 0.
            array_reclasificado = np.where((array >= minimo) & (array <= maximo), array, 0)            
            # Normalizamos para que todas las variables se representen entre 0 y 1.
            array_normalizado = (array_reclasificado - minimo) / (maximo - minimo)
            # Ponderamos el doble a las capas de pendientes superiores a 60º y energía del relieve, 
            # y la mitad al 'mstp' dado que no distingue bien entre cambios bruscos en el relieve 
            # y la cuenca de los ríos. 
            if ruta_raster==ruta_pend or ruta_raster==ruta_po or ruta_raster==ruta_rug:
                peso = 2
            elif ruta_raster==ruta_mstp or ruta_raster==ruta_msrm:
                peso = 0.5
            else:
                peso = 1
 
            if suma_arrays is None:
                suma_arrays = array_normalizado * peso # Primera iteración valida
                
            # Para evitar problemas debido a que el tamaño de los arrays no es el mismo, tomamos el ancho y alto 
            # del primero y lo aplicamos en el resto de arrays. Es una forma de asegurarnos que, en el caso de 
            # que las dimensiones de 'array_normalizado' sean distintas a las de 'suma_arrays', se toman siempre 
            # el mismo ancho y alto para todos los casos. En la siguiente condición añadimos los derivados que 
            # poderan negativamente, es decir, aquellos en los que los valores más bajos son los que se asocian 
            # a zonas donde es posible encontrar elementos aptos para la escalada. 
            elif suma_arrays is not None and ruta_raster in [ruta_svf, ruta_po, ruta_msrm]: 
                filas = min(suma_arrays.shape[0], array_normalizado.shape[0])
                cols  = min(suma_arrays.shape[1], array_normalizado.shape[1])
                suma_arrays = suma_arrays[:filas, :cols]
                array_normalizado = array_normalizado[:filas, :cols]
                suma_arrays -= array_normalizado * peso
                
            else:
                filas = min(suma_arrays.shape[0], array_normalizado.shape[0])
                cols  = min(suma_arrays.shape[1], array_normalizado.shape[1])
                suma_arrays = suma_arrays[:filas, :cols]
                array_normalizado = array_normalizado[:filas, :cols]
                suma_arrays += array_normalizado * peso # Resto de iteraciones
            
            # Dado que haremos una media ponderada no tomamos el número de capas raster sino la suma total 
            # de los pesos:
            raster_validos += peso 

        if suma_arrays is None or raster_validos == 0:
            raise RuntimeError("No se procesó ningún raster válido.")

        media_arrays = suma_arrays/raster_validos
        lito = self.abrir_array_raster (ruta_lito_selec_raster)
        # Nos quedamos con las zonas que presentan litología propiada para la escalada: 
        resultado = media_arrays * lito 
        # Convertimos 0 en 'nodata' ppara que se represente con transparencia en el mapa.
        resultado = np.where(resultado!=0, resultado, np.nan)
        self.alto, self.ancho = resultado.shape

        self.guardar_array(
            resultado,
            ruta_salida)
        
        


# ---
# ### <center><u><b>Clase</b></u> <u><i>menu</i></u></center>
#   
#   En la clase _menu_ creamos un menú en consola que permite al usuario acceder a los distintos cálculos que se realizan en el proyecto e indicar las rutas y nombres de entrada y salida.
# 
# ---

# In[20]:


class menu:
    
    def __init__(self):
        # Las siguientes rutas se inicializan para poder realizar el cálculo final, 
        # que parte de los resultados de las funciones anteriores guardados en disco.
        self.ruta_mdt =  None
        self.ruta_salida = None
        self.ruta_vector = None
        self.ruta_lito = None
        # Incluimos también la clase anterior.
        self.analisis = AnalisisRelieve()
        # Y las opciones del menu:
        self.opciones = {
            '1': self.op1_pend,
            '2': self.op2_pend60,
            '3': self.op3_er,
            '4': self.op4_ld,
            '5': self.op5_svf,
            '6': self.op6_po,
            '7': self.op7_no,
            '8': self.op8_msrm,
            '9': self.op9_mstp,
            '10': self.op10_rug,
            '11': self.op11_lito,
            '12': self.op12_analisis_escalada,
            '13': self.ejec_las2mdt_r
        }

    # Se ha separado la función de 'mostrar_menu' de 'ejecutar_menu' para facilitar la visualización.
    def mostrar_menu(self):
        print('\n --------------------/ Menú /----------------------- \n',
              '1 -  Pendiente \n',
              '2 -  Pendiente superior a 60 grados\n',
              '3 -  Energía del Relieve (ER)\n',
              '4 -  Local Dominance (LD)\n',
              '5 -  Sky View Factor (low) (SVF)\n',
              '6 -  Positive Openness (PO)\n',
              '7 -  Negative Openness (NO)\n',
              '8 -  Multiscale Relief Model (MSRM)\n',
              '9 -  Multiscale Topographic Position (MSTP)\n',
              '10 - Rugosidad\n',
              '11 - Litología\n',
              '12 - Zonas potenciales de escalada\n',
              '13 - Generar MDT a partir de archivos LAS\n',
              '0  - Salir\n')
    
    #-----------------------------------------------------------------------------------------------------------------------------------
    # Con la siguiente función conseguimos mantener el menú abierto en consola hasta que el usuario indique 
    # lo contrario (con la opción 0).
    def ejecutar_menu(self):
        while True:
            self.mostrar_menu()
            opcion = input("Elige una opción: ")
            if opcion == '0':
                print("Ha salido correctamente del menú")
                break # Sale del bucle 'while'
            elif opcion in self.opciones:
                resultado = self.opciones[opcion]()
                if resultado is not None:
                    # Es necesario hacer un segundo display (el primero es el de la función mapa()) para que el resultado
                    # pase entre funciones (de la función mapa() a esta función) y se muestre en la celda. 
                    display(resultado)
            else:
                print("Introduce una opción válida")
    
    #-----------------------------------------------------------------------------------------------------------------------------------
    # La función 'inout' abre ventanas emergentes que permiten al usuario seleccionar el MDT 
    # y el vectorial con las zonas de escalada como entradas, y el directorio de salida.
    def inout(self):
        import tkinter as tk # 'tkinter' es una librería empleada para crear interfaces gráficas.
        from tkinter import filedialog # el submódulo 'filedialog' abre cuadros de diálogo. 

        root = tk.Tk() # Es el objeto principal de 'tkinter', a partir del cual se crea una interfaz gráfica. 
        root.withdraw() # Este método de 'tkinter' oculta la ventana principal.
        # Para abrir una ventana emergente que permita al usuario seleccionar un archivo usamos el submódulo 'askopenfilename'.
        self.ruta_mdt = filedialog.askopenfilename(title='Selecciona el MDT de entrada')
        if not self.ruta_mdt: # En el caso de que el usuario salga de la ventana emergente se imprime un mensaje:
            print('No se ha seleccionado ningún archivo')
        else:
            if self.ruta_mdt.endswith('.tif'):
                pass
            else:
                print('Selecciona un archivo en formato .tif')
                    
        # En este caso se procede de forma similar pero comprobando que el formato del archivo seleccionado es '.shp'
        self.ruta_vector = filedialog.askopenfilename(title='Selecciona el shapefile con las zonas de estudio') 
        if not self.ruta_vector:
            print('No se ha seleccionado ningún archivo')
        else:
            if self.ruta_vector.endswith('.shp'):
                pass
            else: 
                print('Selecciona un archivo en formato shapefile (.shp)')
                self.ruta_vector = filedialog.askopenfilename(title='Selecciona el shapefile con las zonas de estudio') 

        # En este caso, para que el usuario pueda seleccionar una carpeta (directorio de salida) usamos el submódulo 'askdirectory'.
        self.ruta_salida = filedialog.askdirectory(title='Selecciona el directorio de salida')       
        if not self.ruta_salida: 
            print('No se ha seleccionado ningún directorio de salida')
        else:
            pass
        # Devuelve tres rutas, por lo que es necesario indicarlas posteriormente en el mismo orden para evitar errores. 
        return self.ruta_mdt, self.ruta_salida, self.ruta_vector

    #-----------------------------------------------------------------------------------------------------------------------------------

    # Cálculo del MDT a partir de un archivo en formato LAS/LAZ 
    def ejec_las2mdt_r(self,script_ruta):
        # La librería 'subprocess' permite ejecutar comandos como si se hiciera desde consola. 
        import subprocess

        # Comprobamos si existe la ruta al script de R:
        if not os.path.exists(script_ruta):
            raise FileNotFoundError(f"Script R no encontrado: {script_ruta}")
        
        # Indicamos la ruta al ejecutable de R que ejecutará el script:
        rscript_path = r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"
        # En el caso de que la ruta de 'Rscript.exe' sea distinta en la máquina 
        # del usuario devuelve el error tipo 'archivo no encontrado'
        if not os.path.exists(rscript_path):
            raise FileNotFoundError(f"Rscript no encontrado: {rscript_path}")
        
        # Probamos la ejecución del script para controlar los errores sin para el resto de la ejecución.
        try:
            # 'subprocess.run' permite ejecutar comandos desde consola (script en este caso con 
            # el ejcutable 'Rscript.exe'). 
            resultado = subprocess.run(
                [rscript_path, script_ruta],
                capture_output=True, # Guarda la salida en 'resultado' (stdout) y el error en 'e' (stderr)
                text=True,           # Convierte la salida en texto (por defecto son bytes)
                check=True           # Lanza error si se produce un error en la ejecución del script dentro de R.
            )
            print("Salida R:\n", resultado.stdout)
        except subprocess.CalledProcessError as e:
            print("Error en la ejecución de R:")
            print(e.stderr)
            
    #----------------------------------------------############### OPCION 1 DEL MENÚ ###############----------------------------------------
    # El procedimiento que siguen las opciones del menú para los distintos derivados del MDT es similar. En primer lugar, se indica como argumento
    # 'analisis_escalada = False' para que devuelva el mapa con las distintas capas (OSM, sombreado, vectorial con zonas de escalada y raster con 
    # el derivado del MDT en cuestión), y no la ruta (se usa en la última opción para el cálculo de las zonas potenciales de escalada).
    # Luego preguntamos al usuario en consola con 'input' el nombre del archivo de salida. Si no se introduce ningún nombre se asigna uno 
    # por defecto. 
    # Continuamos con la función 'inout' para que el usuario seleccione las rutas de entrada y el directorio de salida. Al directorio de salida
    # le añadimos el nombre introducido anteriormente y el formato (.tif). 
    # Y finalmente ejecutamos el módulo 'mapa' de la clase 'Analisisrelieve' con las rutas anteriores y el nombre del mapa. 
    def op1_pend(self, analisis_escalada = False):
        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para pendientes.\n')
            if nombre == '':
                nombre = 'pendientes'
            else:
                pass
        else:
            nombre = 'pendientes'
            
        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()  
        else: # Si 'analisis_escalada is True' pasamos estas rutas a las variables inicializadas con el mismo nombre para poder usarlas entre funciones. 
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada
        
        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))
        self.analisis.pendiente(ruta_mdt, ruta_salida_archivo)
        print('Cálculo de pendientes finalizado con éxito')
        
        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_salida_archivo, 'Mapa de pendientes')
        else:
            return ruta_salida_archivo

    
    #----------------------------------------------############### OPCION 2 DEL MENÚ ###############----------------------------------------

    def op2_pend60(self, analisis_escalada = False):
        pendiente_valor = 60
        nombre = input(f'Introduce el nombre del archivo de salida para pendientes. El archivo para pendientes superiores a {pendiente_valor} tendrá el mismo nombre y \'_60\'. \n')
        if nombre == '':
            nombre = 'pendientes'
        else:
            pass
        
        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()  
        else: # Si 'analisis_escalada is True' pasamos estas rutas a las variables inicializadas con el mismo nombre para poder usarlas entre funciones. 
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada
        
        ruta_pend = os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))

        self.analisis.pendiente(ruta_mdt, ruta_pend)
        self.analisis.pendiente_selec(ruta_pend, ruta_salida, 
                                      nombre_pendientes = nombre ,
                                      pendiente_valor = pendiente_valor) # Esta en realidad es la de entrada para esta función
        print('Cálculo de pendientes finalizado con éxito')
        ruta_pend60 = os.path.normpath(os.path.join(ruta_salida, f'{nombre}_60.tif'))
        
        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_pend60, 'Mapa de pendientes superiores a 60 grados')
        else:
            return ruta_pend60

    
    #----------------------------------------------############### OPCION 3 DEL MENÚ ###############----------------------------------------
    
    def op3_er(self, analisis_escalada = False):
        if analisis_escalada is False:            
            nombre = input('Introduce el nombre del archivo de salida para Energía del Relieve.\n')
            if nombre == '':
                nombre = 'energia_relieve'
            else:
                pass
        else:
            nombre = 'energia_relieve' 
        
        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()  
        else:
            ruta_mdt = self.ruta_mdt
            ruta_salida = self.ruta_salida
            ruta_escalada = self.ruta_escalada
        
        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))
        self.analisis.energia_relieve(ruta_mdt, ruta_salida_archivo) 
        print('Cálculo de Energia del Relieve finalizado con éxito')

        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_salida_archivo, 
                                          'Energía del Relieve', metodo='media',estad_leyenda='percentil2_98')
        else:
            return ruta_salida_archivo
        

    #----------------------------------------------############### OPCION 4 DEL MENÚ ###############----------------------------------------

    def op4_ld(self, analisis_escalada = False): 
        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para Local Dominance.\n')
            if nombre == '':
                nombre = 'local_dominance'
            else:
                pass
        else:
            nombre = 'local_dominance'

        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()  
        else:
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada 
      
        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))
        self.analisis.herramientas_rvt('ld', ruta_mdt, ruta_salida_archivo) 
        print('Cálculo de \'Local dominance\' finalizado con éxito')

        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_salida_archivo, 'Local Dominance')
        else:
            return ruta_salida_archivo


    #----------------------------------------------############### OPCION 5 DEL MENÚ ###############----------------------------------------

    def op5_svf(self, analisis_escalada = False):
        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para Sky-View Factor.\n')
            if nombre == '':
                nombre = 'sky_view_factor'
            else:
                pass
        else:
            nombre = 'sky_view_factor'

        if analisis_escalada is False: 
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()  
        else:
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada
            
        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))

        self.analisis.herramientas_rvt('svf', ruta_mdt, ruta_salida_archivo)  
        print('Cálculo de \'Sky-View Factor (low)\' finalizado con éxito')

        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_salida_archivo, 'Sky-View Factor')
        else:
            return ruta_salida_archivo
            

    #----------------------------------------------############### OPCION 6 DEL MENÚ ###############----------------------------------------

    def op6_po(self, analisis_escalada = False):
        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para Positive Openness.\n')
            if nombre == '':
                nombre = 'positive_openness'
            else:
                pass
        else:
            nombre = 'positive_openness'

        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()  
        else:
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada
        
        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))
        
        self.analisis.herramientas_rvt('po', ruta_mdt, ruta_salida_archivo)  
        print('Cálculo de \'Positive Openness\' finalizado con éxito')
        
        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_salida_archivo,'Positive Openness', estad_leyenda = 'percentil2_98')
        else:
            return ruta_salida_archivo

    #----------------------------------------------############### OPCION 7 DEL MENÚ ###############----------------------------------------
    
    def op7_no(self, analisis_escalada = False):
        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para Negative Openness.\n')
            if nombre == '':
                nombre = 'negative_openness'
            else:
                pass
        else:
            nombre = 'negative_openness'
            
        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()  
        else:
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada
            
        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))
        
        self.analisis.herramientas_rvt('no', ruta_mdt, ruta_salida_archivo) 
        print('Cálculo de \'Negative Openness\' finalizado con éxito')
        
        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_salida_archivo, 'Negative Openness', estad_leyenda = 'percentil2_98')
        else:
            return ruta_salida_archivo

    
    #----------------------------------------------############### OPCION 8 DEL MENÚ ###############----------------------------------------
            
    def op8_msrm(self, analisis_escalada = False):
        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para Multiscale Relief Model.\n')
            if nombre == '':
                nombre = 'multiscale_relief'
            else:
                pass        
        else:
            nombre = 'multiscale_relief'
        
        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()
        else:
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada
            
        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))

        self.analisis.herramientas_rvt('msrm', ruta_mdt, ruta_salida_archivo)  
        print('Cálculo de \'Multiscale Relief Model\' finalizado con éxito')

        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_salida_archivo, 'Multiscale Relief Model', estad_leyenda='percentil2_98')
        else:
            return ruta_salida_archivo


    #----------------------------------------------############### OPCION 9 DEL MENÚ ###############----------------------------------------
            
    def op9_mstp(self, analisis_escalada = False):
        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para Multiscale Topographic Position.\n')
            if nombre == '':
                nombre = 'multiscale_topo'
            else:
                pass
        else:
            nombre = 'multiscale_topo'

        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()
        else:
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada

        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))

        self.analisis.herramientas_rvt('mstp', ruta_mdt, ruta_salida_archivo)  
        print('Cálculo de \'Posición topográfica multiescala\' finalizado con éxito')

        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_salida_archivo, 'Multiscale Topographic Position')
        else:
            return ruta_salida_archivo

    
    #----------------------------------------------############### OPCION 10 DEL MENÚ ###############----------------------------------------
            
    def op10_rug(self, analisis_escalada = False):
        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para Rugosidad.\n')
            if nombre == '':
                nombre = 'rugosidad'
            else:
                pass
        else: 
            nombre = 'rugosidad'

        if analisis_escalada is False:
            ruta_mdt, ruta_salida, ruta_escalada = self.inout()  
        else:
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada
            
        ruta_rug = os.path.normpath(os.path.join(ruta_salida,f'{nombre}.tif'))
        self.analisis.rugosidad_logn (ruta_mdt, ruta_rug)
        print('Cálculo de \'Rugosidad\' finalizado con éxito')
        ruta_rug_ln = os.path.normpath(os.path.join(ruta_salida,f'{nombre}_ln.tif'))
        
        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, ruta_rug_ln, 'Rugosidad')
        else:
            return ruta_rug_ln

    
    #----------------------------------------------############### OPCION 11 DEL MENÚ ###############----------------------------------------
            
    def op11_lito(self, analisis_escalada = False):
        import tkinter as tk
        from tkinter import filedialog

        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para litología.\n')
            if nombre == '':
                nombre = 'lito_sel'
            else:
                pass
        else:
            nombre = 'lito_sel'

        ruta_lito = filedialog.askopenfilename(title='Selecciona el archivo de unidades geológicas (shapefile .shp)') 
        
        if analisis_escalada is False:
            ruta_mdt = filedialog.askopenfilename(title='Selecciona el MDT de entrada')  
            ruta_escalada = filedialog.askopenfilename(title='Selecciona el archivo de zonas de escalada (shapefile .shp)') 
            ruta_salida = filedialog.askdirectory(title='Selecciona el directorio de salida')      
        else: 
            ruta_mdt = self.ruta_mdt; ruta_salida = self.ruta_salida; ruta_escalada = self.ruta_escalada
        
        ruta_salida_archivo =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.shp'))
            
        self.analisis.procesado_vector(ruta_lito, 'UnidadGeol', ruta_salida_archivo)
        print('\'Proceso de disolución y selección de litología característica en escalada\' finalizado con éxito')
        
        if analisis_escalada is False:
            return AnalisisRelieve().mapa(ruta_mdt, ruta_escalada, 'Unidades geológicas', ruta_lito= ruta_salida_archivo)
        else:
            return ruta_salida_archivo
        

    #----------------------------------------------############### OPCION 12 DEL MENÚ ###############----------------------------------------
    # En este método el usuario selecciona el vectorial con las unidades litológicas y el directorio de salida, y ejecuta los métodos 
    # 'procesado_vector' y 'rasterizar_vectorial' de la clase anterior, para generar finalmente un raster con valores 1 (litología  
    # apropiada) y 0 (resto de zonas), que se usa en el siguiente método. 
    def rasterizar_lito_selec(self, analisis_escalada = False):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw() 

        if analisis_escalada is False:
            nombre = input('Introduce el nombre del archivo de salida para litología.\n')
            if nombre == '':
                nombre = 'lito_sel_bool'
            else:
                pass
        else:
            nombre = 'lito_sel_bool'
        
        ruta_lito = filedialog.askopenfilename(title='Selecciona el shapefile de litología')

        if analisis_escalada is False:
            ruta_salida = filedialog.askdirectory(title='Selecciona el directorio de salida')
        else:
            ruta_salida = self.ruta_salida
            
        ruta_salida_lito =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.shp'))
        ruta_resultado_lito = self.analisis.procesado_vector(ruta_lito, 'UnidadGeol', ruta_salida_lito)
            
        ruta_salida_raster =  os.path.normpath(os.path.join(ruta_salida, f'{nombre}.tif'))
        ruta_lito_selec_raster = self.analisis.rasterizar_vectorial(ruta_resultado_lito, ruta_salida_raster, valor=1)
        return ruta_lito_selec_raster

    # En la última opción se ejecutan las anteriores y finalmente se llama al método 'analisis_escalada' de la clase 
    # 'Analisisrelieve' para calcular las zonas potenciales de escalada. 
    # En este caso se usa el argumento 'analisis_escalada=True' para devolver la ruta con el resultado y no el mapa. 
    def op12_analisis_escalada(self):
        self.ruta_mdt, self.ruta_salida, self.ruta_escalada = self.inout()
        
        ruta_pend = self.op1_pend(analisis_escalada=True)
        ruta_er = self.op3_er(analisis_escalada=True)
        ruta_ld = self.op4_ld(analisis_escalada=True)
        ruta_svf = self.op5_svf(analisis_escalada=True)
        ruta_po = self.op6_po(analisis_escalada=True)
        ruta_msrm = self.op8_msrm(analisis_escalada=True)
        ruta_mstp = self.op9_mstp(analisis_escalada=True)
        ruta_rug = self.op10_rug(analisis_escalada=True)
        ruta_lito_selec_raster = self.rasterizar_lito_selec(analisis_escalada=True)

        nombre = input('Introduce el nombre del archivo de salida para el raster con las zonas potenciales de escalada.\n')
        if nombre == '':
            nombre = 'Zonas_Potenciales_Escalada'
        else:
            pass

        ruta_salida_archivo =  os.path.normpath(os.path.join(self.ruta_salida, f'{nombre}.tif'))
        
        self.analisis.analisis_escalada (self.ruta_escalada, ruta_pend, ruta_er, 
                                       ruta_ld, ruta_svf, ruta_po, ruta_msrm, ruta_mstp, 
                                       ruta_rug, ruta_lito_selec_raster, ruta_salida_archivo)

        print('Los cálculos han finalizado correctamente y se ha generado el resultado con las zonas potenciales de escalada.')
        
        return AnalisisRelieve().mapa(self.ruta_mdt, self.ruta_escalada, ruta_salida_archivo, 'Zonas Potenciales de Escalada') 
        

    #----------------------------------------------############### EJECUCIÓN DEL MENÚ ###############----------------------------------------
    # Para facilitar la ejecución del menú, usamos un método que instancia la clase y llama automáticamente a 'ejecutar_menu'.
    @classmethod #-->  Es un decorador que indica que el método 'run' es un método de clase, es decir, que como argumento tiene
                 #     'cls' (la clase) en lugar de 'self' (la instancia).
    def run(cls):
        ejecucion_menu = cls()
        
        ejecucion_menu.ejecutar_menu()




menu = menu()
menu.ejecutar_menu()


# ---
# ### *Bibliografía*
# -	Sánchez-Fernández, M.; Arenas-García, L.; Gutiérrez Gallego, J.A. Detection of Construction and Demolition Illegal Waste Using Photointerpretation of DEM Models of LiDAR Data. _Land_ **2023**, 12, 2119. https://doi.org/10.3390/land12122119
# 
# -	Zakšek, K.; Oštir, K.; Kokalj, Ž. Sky-View Factor as a Relief Visualization Technique. _Remote Sensing_. **2011**, 3, 398-415. https://doi.org/10.3390/rs3020398
# 
# -	Zaksek, Klemen & Oštir, Krištof & Kokalj, Žiga. Sky-View Factor as a Relief Visualization Technique. _Remote Sensing_. **2011**, 3. https://doi.org/10.3390/rs3020398.
# 
# -	Yokoyama, R., Shirasawa, M., & Pike, R. J. Visualizing Topography by Openness: A New Application of Image Processing to Digital Elevation Models. _Photogrammetric Engineering and Remote Sensing_. **2002**, 68, 257-265.
# 
# -	Hesse, R., Lieberwirth, U. & Herzog, I. Visualisierung hochauflösender Digitaler Geländemodelle mit LiVT. In Computeranwendungen und Quantitative Methoden in der Archäologie. 4. _Berlin Studies of the Ancient World, Edition Topoi_. **2016**, 109–128.
# 
# -	Lindsay, John & Cockburn, Jaclyn & Russell, Hazen. An Integral Image Approach to Performing Multi-Scale Topographic Position Analysis. _Geomorphology_. **2015**, 245, 51-61. https://doi.org/10.1016/j.geomorph.2015.05.025.
# 
# -	Gallant, J.C., Wilson, J.P. Primary topographic attributes. _Terrain Analysis: Principles and Applications. John Wiley & Sons Inc., New York_. **2000**, 51–85.
# 
# -	Weiss, A. Topographic position and landforms analysis (Poster Presentation).  _ESRI User Conference, San Diego, CA_ . **2001**, 200. 
# 
# -	Orengo, Hector & Petrie, Cameron. Multi-Scale Relief Model (MSRM): a new algorithm for the visualisation of subtle topographic change of variable size in digital elevation models. _Earth Surface Processes and Landforms_. **2017**, 43. https://doi.org/10.1002/esp.4317.
# 
# -	Wilson, M. F. J., O’Connell, B., Brown, C., Guinan, J. C., & Grehan, A. J. Multiscale Terrain Analysis of Multibeam Bathymetry Data for Habitat Mapping on the Continental Slope. _Marine Geodesy_. **2007**, 30(1–2), 3–35. https://doi.org/10.1080/01490410701295962
# 
# 
