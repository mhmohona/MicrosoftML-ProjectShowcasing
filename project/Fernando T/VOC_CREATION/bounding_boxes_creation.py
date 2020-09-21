## Create dataset
import cv2
import glob
import os
from tqdm import tqdm
import argparse

# sorter
from operator import itemgetter
from pascal_voc_writer import Writer


class ImageHanlder:
    def __init__(self, draw_rects=False,  fixed=False, base_path="results/"):
        """
        This class is used for create the VOC like dataset using computer vision techniques.

        """
        self.font                   = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale              = 0.5
        self.fontColor              = (255,0,255)
        self.lineType               = 2

        # Bounding rectangles placholder
        self.outputs = []


        # Placholder for the partidos.
        self.partidos_todos = []
        self.partidos = []

        # Expected position of partidos in 1088, 872 Resolution image
        # in the acta region.

        # CC Name box
        # P1 : 14,40
        # P2 : 520,90
        
        # Region containing the information about the "Mesas" to count
        self.P1 = (90*4, 98*4)
        self.P2 = (272*4, 218*4)
        
        
        self.draw_rects = draw_rects
        
        # Values Right
        partidos_d = [24, 22, 20, 18, 16, 14, 12, 10, 8, 6]
        
        # Values Left
        partidos = [23, 21, 19, 17, 15, 13, 11, 9, 7]

        otros_d = [4, 2, 6]
        otros = [5, 3, 1]

        self.partidos_todos = partidos + partidos_d

        self.otros_todos = otros + otros_d

        self.ids = partidos + partidos_d + otros + otros_d
        

        self.base_path = base_path

        self.fixed = fixed
        
        # Normal resolution to resize
        self.normal_shape = (2500, 1600)

        # Create mapping IDs
        self.name_maping = {
            1: "Votos Nulos Presidente",
            2: "Votos Nulos Diputado",
            3: "Votos Blancos Presidente",
            4: "Votos Blancos Diputado",
            5: "Votos Validos Presidente",
            6: "Votos Validos Diputado",
            7: "PAN-BOL Presidente",
            8: "PAN-BOL Diputado",
            9: "MNR Presidente",
            10: "MNR Diputado",
            11: "PDC Presidente",
            12: "PDC Diputado",
            13: "21F Presidente",
            14: "21F Diputado",
            15: "MAS IPSP Presidente",
            16: "MAS IPSP Diputado",
            17: "UCS Presidente" ,
            18: "UCS Diputado",
            19: "MTS Presidente",
            20: "MTS Diputado",
            21: "FPV Presidente",
            22: "FPV Diputado",
            23: "CC Presidente",
            24: "CC Diputado"
        }

        self.name_maping_results = {
            25: "Votos Nulos Presidente, Result",
            26: "Votos Nulos Diputado, Result",
            27: "Votos Blancos Presidente, Result",
            28: "Votos Blancos Diputado, Result",
            29: "Votos Validos Presidente, Result",
            30: "Votos Validos Diputado, Result",
            31: "PAN-BOL Presidente, Result",
            32: "PAN-BOL Diputado, Result",
            33: "MNR Presidente, Result",
            34: "MNR Diputado, Result",
            35: "PDC Presidente, Result",
            36: "PDC Diputado, Result",
            37: "21F Presidente, Result",
            38: "21F Diputado, Result",
            39: "MAS IPSP Presidente, Result",
            40: "MAS IPSP Diputado, Result",
            41: "UCS Presidente, Result" ,
            42: "UCS Diputado, Result",
            43: "MTS Presidente, Result",
            44: "MTS Diputado, Result",
            45: "FPV Presidente, Result",
            46: "FPV Diputado, Result",
            47: "CC Presidente, Result",
            48: "CC Diputado, Result"
        }


    def write_row_debug_log(self, file_name, path_img, label):
        """
        Write log
        """
        os.makedirs(self.base_path, exist_ok=True)

        path = f"{self.base_path}/{file_name}.txt"

        with open(path,"a+") as f:
            f.write(f"{path_img},{label} \n")


    def write_row_results_log(self, file_name, image_name, counts, id):
        """
        Write the last results
        """
        os.makedirs(self.base_path, exist_ok=True)
        path = f"{self.base_path}/{file_name}.txt"

        with open(path,"a+") as f:
            f.write(f"{image_name},{counts},{id} \n")


    def norm_image(self, image):
        """
        Norm image to constant shape
        """
        img = cv2.resize(image, self.normal_shape)
        return img

    def cut_image(self, image, p1, p2, simple=False):
        """
        Crop any imagen given P1 and P2
        """
        x = p1[0]
        y = p1[1]

        w = p2[0]
        h = p2[1]
        
        if simple:
            crop_img = image[y:h, x:w]
        crop_img = image[y:y+h, x:x+w]
        return crop_img

    def find_contour_base(self, image, kernel):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        
        #--- performing Otsu threshold ---
        ret,thresh1 = cv2.threshold(gray, 0, 255, 
                                    cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
        #--- choosing the right kernel
        #--- kernel size of 3 rows (to join dots above letters 'i' and 'j')
        #--- and 10 columns to join neighboring letters in words and neighboring words
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        #---Finding contours ---
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        im2 = image.copy()

        return im2, contours

    def find_contour(self, image, image_path):
        #(30, 2) for letters
        im2, contours = self.find_contour_base(image, (7,2))

        boxes = []
        outputs =[]

        # Iterate over all contours
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            p1 = (x, y)
            p2 = (x + w, y + h)

            deltax = p2[0]-p1[0]
            deltay = p2[1]-p1[1]

            # in range? of a box?
            if (200<=(deltax)<=360) and (30<=(deltay)<=70):
                # print(deltax)
                # print(deltay)
                p1 = (p1[0], p1[1])
                p2 = (p2[0], p2[1])

                boxes.append([p1,p2])
                
                #cv2.rectangle(im2, p1, p2, (0, 255, 0), 2)

        
        # If all the rectanguler boxes was found.
        #thresh = ((len(boxes) >= 20 ) and (len(boxes) <= 24))
        thresh = len(boxes) == 24

        if thresh :
            # print("size",size)

            # Iterate over the rectangular boxes
            for i, b in enumerate(boxes):
                index_or = i

                out_index = index_or + 1

                outputs.append({out_index: b})
        else:
            """
            If detection is < 24 log this file
            """
            file_name = "detect_boxes_error_log"
            self.write_row_debug_log(file_name, image_path, thresh)

        return im2, outputs
    
    
    def main(self, images_list):
        
        for i_path in tqdm(images_list, ascii=True, desc="Reading..."):
            try:
                #i = "actas/mesas/200081.jpg"
                img = cv2.imread(i_path)

                if self.fixed:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img_r   =   self.norm_image(img)
                # extract only the votation box
                c_image =   self.cut_image(img_r, self.P1, self.P2)

                # Obtain the full filename
                filename = i_path.split("/")[-1]

                # Obtain only the name without extension 
                filename_root = filename.split(".")[0]
                
                # Find the contours box rectangle for each partido 
                cont_img, self.outputs = self.find_contour(c_image, i_path)
                
                if len(self.outputs) > 0:
                        
                    _writer = Writer(i_path, self.normal_shape[1], self.normal_shape[0])

                    for o in self.outputs:
                        # Iterate over the rectangular boxes
                        for k, v in o.items():
                            #print(f"WORKING ON {k}")
                            partido_key_id = k  
                            
                            p1, p2 = v

                            if k in self.partidos_todos:
                                # Mueva Position tira 
                                n_p1 = (p1[0] + 340, p1[1])
                                n_p2 = (p2[0] + 180, p2[1])

                            if k in self.otros_todos:

                                # Mueva Position tira 
                                n_p1 = (p1[0] + 310, p1[1])
                                n_p2 = (p2[0] + 215, p2[1])

                            # Names
                            xmin = p1[0] 
                            ymin = p1[1] 
                            xmax = p2[0] 
                            ymax = p2[1]

                            # Results
                            xmin_n = n_p1[0] 
                            ymin_n = n_p1[1] 

                            xmax_n = n_p2[0] 
                            ymax_n = n_p2[1] 


                            # ::addObject(name, xmin, ymin, xmax, ymax)
                            # Get names labels
                            label = self.name_maping[int(partido_key_id)]

                            # Get Names results lables
                            label_n = self.name_maping_results[int(partido_key_id+24)]
                            
                            
                            # Write coordinates For Names

                            _writer.addObject(label, xmin, ymin, 
                                                    xmax, ymax)

                            # Write coordinates for  Results
                            _writer.addObject(label_n, xmin_n, ymin_n, 
                                                    xmax_n, ymax_n)


                            if self.draw_rects:

                                p1 =  ( xmin, ymin )
                                p2 =  ( xmax, ymax )
                                
                                p1_n = (xmin_n, ymin_n)
                                p2_n = (xmax_n, ymax_n)
                                
                                font                   = cv2.FONT_HERSHEY_SIMPLEX
                                fontScale              = 0.6
                                fontColor              = (0,0,0)
                                lineType               = 1

                                cv2.putText(cont_img, label, 
                                        ( xmin, ymin  +50), 
                                        font, 
                                        fontScale,
                                        fontColor,
                                        lineType)

                                cv2.putText(cont_img, label_n, 
                                        (xmin_n, ymin_n +50), 
                                        font, 
                                        fontScale,
                                        fontColor,
                                        lineType)

                                # Draw names
                                cv2.rectangle(cont_img, p1, p2, (0, 255, 0), thickness=2)

                                # Draw Results
                                cv2.rectangle(cont_img,  p1_n , p2_n, (0, 255, 0), thickness=2)

                            
                    # Write the acta with the number of results drawed on it.
                    base_path_labels = os.path.join(self.base_path,"Train","annotations")
                    base_path_images =  os.path.join(self.base_path, "Train", "images")

                    os.makedirs(base_path_labels, exist_ok=True)
                    os.makedirs(base_path_images, exist_ok=True)

                    path_label= os.path.join(base_path_labels, filename_root+".xml" )
                    path_image= os.path.join(base_path_images, filename_root+".jpg")

                    
                    # Write XML
                    _writer.save(path_label)

                    cv2.imwrite(path_image, cont_img)
                
            except Exception as e:
                filename = i_path.split("/")[-1]
                _filename_error = "errorInOpenFile"
                self.write_row_debug_log(_filename_error, filename, e)
                print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Actas de mesa to counts')

    parser.add_argument('--data_path', type=str, default="images/", help='Where is your jpg image data. absolute path')
    parser.add_argument('--data_output_path', type=str, default="results/", help='Where TO PUT your new jpg image data. absolute path')
    
    args = parser.parse_args()

    images_path = args.data_path
    data_output_path = args.data_output_path

    os.makedirs(data_output_path, exist_ok=True)

    # Images lists...
    images_list = glob.glob(f"{images_path}*.jpg")

    image_hanlder = ImageHanlder(
                                draw_rects=False,
                                base_path = data_output_path
                                ) 

    image_hanlder.main(
        images_list
    )
