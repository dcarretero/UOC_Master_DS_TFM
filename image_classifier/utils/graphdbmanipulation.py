import os
import pandas as pd
import networkx as nx


class ImageAnalyzed:
    def __init__(self, name):
        self.name = name
        self.objects = {}

    def add_object(self, object_class):
        if object_class in self.objects.keys():
            self.objects[object_class] = self.objects[object_class] + 1
        else:
            self.objects[object_class] = 1


class ImagesGraphDB:
    def __init__(self):
        self.graph = nx.Graph()
        self.images_analyzed = []

    def _load_extracted_labels_data(self,txt_files_dir,labels):
        for file_txt in os.listdir(txt_files_dir):
            if os.path.isfile(os.path.join(txt_files_dir, file_txt)):
                with open(os.path.join(txt_files_dir, file_txt)) as f:
                    lines = f.readlines()
                    image_name = file_txt.split('.')[0]
                    imageAnalyzed = ImageAnalyzed(image_name+'.jpg')
                    for line in lines:
                        line= line.replace('\n','')
                        line_splitted = line.split(' ')
                        imageAnalyzed.add_object(labels[int(line_splitted[0])])
            self.images_analyzed.append(imageAnalyzed)

    def _load_graph_from_images_analyzed(self):
        graph = nx.Graph()
        for image in self.images_analyzed:
            atrib_image_node = {}
            atrib_image_node['type'] = 'image_filename'
            graph.add_nodes_from([(image.name,atrib_image_node)])
            for object_class in image.objects.keys():
                if not self.graph.has_node(object_class):
                    atrib_class_node = {}
                    atrib_class_node['type'] = 'object_class'
                    graph.add_nodes_from([(object_class,atrib_class_node)])
                graph.add_edges_from([(image.name,object_class)],weight=image.objects[object_class])
        self.graph = nx.compose(self.graph,graph) #fusiona el nuevo grafo con el existente

    # función que cruza dos listas y devuelve la intersección de elementos
    def _intersection(self,lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    def get_images_containing_list_object_types(self,object_types):
        nodes = nx.nodes(self.graph)  # devuelve un diccionario iterable
        counter=0
        neighbors=[]
        for node in nodes:
            if node in object_types:
                if counter==0:
                    neighbors = list(nx.neighbors(self.graph, node))
                else:
                    neighbors= self._intersection(neighbors,list(nx.neighbors(self.graph, node)))
                counter+=1
        return neighbors

    def get_nodes(self):
        return nx.nodes(self.graph)
    def get_neighbours(self,node):
        return nx.neighbors(self.graph,node)
    def get_graph(self):
        return self.graph
    def add_node(self,node):
        self.graph.add_node(node)
    def add_edge(self,node_from, node_to):
        self.graph.add_edge(node_from,node_to)
    def set_attribute_to_node(self,node,attribute,value):
        self.graph.nodes[node][attribute] = value
    def set_attribute_to_edge(self,node_from,node_to,attribute,value):
        self.graph.edges[node_from,node_to][attribute] = value
    def get_attribute_value_node(self,node,attribute):
        return self.graph.nodes[node][attribute]
    def get_attribute_value_edge(self,node_from,node_to,attribute,value):
        return self.graph.edges[node_from, node_to][attribute]
    def write_gml_file(self,path_file_gml):
        nx.write_gml(self.graph,path_file_gml)
    def load_graph_from_gml_file(self,path_file_gml):
        graph = nx.read_gml(path_file_gml)
        self.graph=nx.compose(self.graph,graph)
    def load_graph_from_yolo_detected_objects_txt_files(self, labels_reference_file,txt_files_dir):
        labels = pd.read_csv(labels_reference_file, header=None, names=['label']).label.tolist()
        self._load_extracted_labels_data(txt_files_dir,labels)
        self._load_graph_from_images_analyzed()




