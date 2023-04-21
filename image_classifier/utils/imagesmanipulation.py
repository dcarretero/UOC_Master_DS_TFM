from PIL import Image
import os
from glob import glob
import pandas as pd
from functools import reduce
from xml.etree import ElementTree as et
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
import textwrap, os
from GPSPhoto import gpsphoto
from exif import Image as ExifImage
from datetime import datetime
import requests
import h3
import face_recognition as fc
import cv2
import matplotlib.pyplot as plt

class ImageHelper:

    def _save_txt(self, df_grouped,output_path):
        for filename, data in df_grouped:
            txt_filename = os.path.splitext(filename)[0]
            path_txt_filename = output_path + '/' + txt_filename +'.txt'
            df_grouped.get_group(filename).set_index('filename').to_csv(path_txt_filename,sep=' ',index= False, header= False)

    def resize_images(self,input_dir_path,output_subdir,width,height):
        #Se crea el subdirectorio si no existe
        os.makedirs(os.path.join(input_dir_path, output_subdir), exist_ok=True)
        for path in os.listdir(input_dir_path):
            if os.path.isfile(os.path.join(input_dir_path, path)):
                split_tup = os.path.splitext(path)
                file_extension = split_tup[1]
                file_name = split_tup[0]
                if file_extension.find('xml') == -1:
                    image = Image.open(os.path.join(input_dir_path, path))
                    if image.size[0] > width and image.size[1] > height:
                        new_image = image.resize((width, height))
                        new_file_name = file_name  + file_extension
                        new_image.save(os.path.join(input_dir_path,output_subdir, new_file_name))

    def generate_txt_label_files(self,labels,input_dir_path,output_subdir):
        # Se fija el criterio de busqueda de ficheros con *.xml en el directorio de imagenes redimensionadas
        search_criteria =str(os.path.join(input_dir_path,output_subdir)).replace('\\','/')+'/*.xml'
        # Se obtienen los nombres de los ficheros
        xml_list = glob(search_criteria)
        # Se cambian los separadores de ruta
        xml_list = list(map(lambda x:x.replace('\\','/'),xml_list))
        parser = []
        for xml in xml_list:

            tree = et.parse(xml)    # Se obtiene el árbol xml
            root = tree.getroot()   # Se obtiene el nodo raiz
            image_name = root.find('filename').text  # Se obtiene el nombre de la imagen informado en el fichero
            width = root.find('size').find('width').text # Se obtiene el ancho de la imagen en pixels
            height = root.find('size').find('height').text # Se obtiene el alto de la imagen en pixels
            objs = root.findall('object') # Se encuentran todos los objetos etiquetados en la imagen
            for obj in objs: # Se recorren los objetos para obtener el nombre y las dimensiones del bounding box
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin= bndbox.find('xmin').text
                xmax= bndbox.find('xmax').text
                ymin= bndbox.find('ymin').text
                ymax= bndbox.find('ymax').text
                parser.append([image_name,width,height,name,xmin,xmax,ymin,ymax])
        # Se crea un pandas dataframe a partir de la información en la lista parser
        df = pd.DataFrame(parser, columns=['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])
        # Se calculan las columnas necesarias en etiquetas utilizables por la libreria Yolo
        cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
        df[cols] = df[cols].astype(int)
        df['center_x'] = ((df['xmax'] + df['xmin']) / 2) / df['width']
        df['center_y'] = ((df['ymax'] + df['ymin']) / 2) / df['height']
        df['w'] = (df['xmax'] - df['xmin']) / df['width']
        df['h'] = (df['ymax'] - df['ymin']) / df['height']
        df['id'] = df['name'].apply(lambda x: labels[x])
        cols =['filename','id','center_x','center_y','w','h']
        # Se agrupan por fichero
        df_yolo = df[cols].groupby('filename')
        self._save_txt(df_yolo,os.path.join(input_dir_path,output_subdir))

    def display_images(
            images: [Image],
            columns=5, width=20, height=8, max_images=15,
            label_wrap_length=50, label_font_size=8):

        if not images:
            print("No images to display.")
            return

        if len(images) > max_images:
            print(f"Showing {max_images} images of {len(images)}:")
            images = images[0:max_images]

        height = max(height, int(len(images) / columns) * height)
        plt.figure(figsize=(width, height))
        for i, image in enumerate(images):

            plt.subplot(int(len(images) / columns + 1), columns, i + 1)
            plt.imshow(image)

            if hasattr(image, 'filename'):
                title = image.filename
                if title.endswith("/"): title = title[0:-1]
                title = os.path.basename(title)
                title = textwrap.wrap(title, label_wrap_length)
                title = "\n".join(title)
                plt.title(title, fontsize=label_font_size);
    def face_recognition(self,face_original_image_encodings,image_to_analyze_path):
        # Localizamos las caras de la imagen a comparar
        image_to_compare = cv2.imread(image_to_analyze_path, cv2.IMREAD_COLOR)
        face_locations = fc.face_locations(image_to_compare)
        if face_locations != []:
            for face_location in face_locations:
                face_image_compared_encodings = fc.face_encodings(image_to_compare, known_face_locations=[face_location])[0]
                result = fc.compare_faces([face_image_compared_encodings], face_original_image_encodings)
                if result[0] == True:
                    return (True,face_location[0],face_location[1],face_location[2],face_location[3])
        return (False,0,0,0,0)

    def generate_images_dir_hiperlinks_csv(self,images_path, output_path,filename):
        entries = os.listdir(images_path)
        df_entries = pd.DataFrame(entries,columns=['filename'])
        df_entries['url']=df_entries['filename'].apply(lambda x: '=HIPERVINCULO(\"'+ images_path+x+'\";\"'+x+'\")')
        path_csv_filename = os.path.join(output_path, filename)
        df_entries['url'].to_csv(path_csv_filename, sep=',', index=False, header=False,encoding='utf-8-sig')

    def get_geo_and_date_images_data(self,images_path,residence_latitude, residence_longitude):

        entries = os.listdir(images_path)
        geo_dates_data=[]
        for entry in entries:
            if not os.path.isfile(os.path.join(images_path,entry)):
                # Para saltar directorios
                continue
            gps_data = gpsphoto.getGPSData(os.path.join(images_path,entry))
            try:
                latitude = gps_data.get('Latitude',None)
                longitude = gps_data.get('Longitude',None)
                if (latitude is None and longitude is None):
                    city =  'unknown'
                    postcode = 'unknown'
                    state_district = 'unknown'
                    state = 'unknown'
                    country = 'unknown'
                    distance_km =None
                else:
                    params ={
                        'format':'geojson',
                        'lat': latitude,
                        'lon':longitude
                    }
                    resp = requests.get("https://nominatim.openstreetmap.org/reverse", params=params)
                    respjson = resp.json()
                    address = respjson['features'][0]['properties']['address']
                    city = address.get('city','unknown')
                    postcode = address.get('postcode','unknown')
                    state_district = address.get('state_district','unknown')
                    state = address.get('state','unknown')
                    country = address.get('country','unknown')
                    distance_km = h3.point_dist((residence_latitude,residence_longitude),(latitude,longitude),unit='km')

                with open(os.path.join(images_path,entry), 'rb') as src:
                    img = ExifImage(src)
                    datestring=img.get("datetime")
                    date = datetime.strptime(datestring,'%Y:%m:%d %H:%M:%S')
                    year = date.year
                    month = date.month
                    day = date.day
                data_record = [entry,latitude,longitude,city,postcode,state_district,state,country,distance_km,datestring,day,month,year]
                geo_dates_data.append(data_record)
            except Exception as e:
                print(f"Error en {entry} : {e}")
        df_geo_dates_data = pd.DataFrame(geo_dates_data, columns=['filename','latitude','longitude','city','postcode','state_district',
                                                    'state','country','distance_km','datestring','day','month','year'])

        return df_geo_dates_data







