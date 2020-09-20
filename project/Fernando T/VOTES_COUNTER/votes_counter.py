import cv2
import os
import sys
from pathlib import Path 
import glob
import argparse
from tqdm import tqdm
import base64
import json
import requests

CURRENT_PROJECT_PATH = os.getcwd()
print(str(Path(CURRENT_PROJECT_PATH).parents[0]))


sys.path.insert(1, str(Path(CURRENT_PROJECT_PATH).parents[0]))

from VOC_CREATION.bounding_boxes_creation import ImageHanlder


class VotesCounter(ImageHanlder):
    def __init__(self, draw_results=True, cut_numbers=False):
        super().__init__()

        self.cut_numbers = cut_numbers

        # Bounding rectangles placholder
        self.outputs = []

        # For handle the data 
        self.data_handler = []

        # Placholder for the partidos.
        self.partidos_todos = []
        self.partidos = []

        # If draw results
        self.draw_results = draw_results
     
        # Expected position of partidos in 1088, 872 Resolution image
        # in the acta region.

        # CC Name box
        # P1 : 14,40
        # P2 : 520,90
        
        # Region containing the information about the Mesas to count
        self.P1 = (90*4, 98*4)
        self.P2 = (272*4, 218*4)
        
            
        partidos_d = [24, 22, 20, 18, 16, 14, 12, 10, 8, 6]
        partidos = [23, 21, 19, 17, 15, 13, 11, 9, 7]

        otros_d = [4, 2, 6]
        otros = [5, 3, 1]

        self.partidos_todos = partidos + partidos_d

        self.otros_todos = otros + otros_d

        self.ids = partidos + partidos_d   + otros + otros_d
        
        self.base_path = "results_votes/"
        
        self.all_names_dict = {**self.name_maping, **self.name_maping_results}

    def call_server_prediction(self, numpy_img):
        """
        Call Instance of Custom MNIST model

        """
        # image path
        MODEL_REST_API_URL = "http://localhost:8501/analyze:predict"

        #numpy_img = Image.open(IMAGE_PATH)

        #numpy_img = cv2.imread(IMAGE_PATH)
        # _, numpy_img_encoded = cv2.imencode('.jpg', numpy_img)

        #numpy_img_encoded = base64.b64encode(numpy_img)
    
        _, buffer = cv2.imencode('.jpg', numpy_img)
        numpy_img_encoded_as_text = base64.b64encode(buffer)
        
        numpy_img_encoded = numpy_img_encoded_as_text.decode("utf-8")

        payload = {
            "instances": [
                {
                "image_bytes": {
                    "b64": f"{numpy_img_encoded}"
                }
                }
            ],
        }

        payload = json.dumps(payload)

        r = requests.post(MODEL_REST_API_URL, data=payload).json()
        # loop over the predictions and display them
        return r["predictions"]

    def draw_rectangle_numpy(self, img, p1,p2):
        """
        Draw reactangle 
        """

        img = cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        return img

    def main(self, images_list):
        """
        Expected input, full image 
        """

        for i_path in tqdm(images_list[:5], ascii=True, desc="Reading..."):
            print(f"IMAGE : {i_path}")
            self.interchange = False

            #try:
            # Try to open the image
            #i = "actas/mesas/200081.jpg"
            img     =   cv2.imread(i_path)

            img_r   =   self.norm_image(img)

            # extract only the votation box
            c_image =   self.cut_image(img_r, self.P1, self.P2)

            # Obtain the full filename
            filename = i_path.split("/")[-1]

            # Obtain only the name without extension 
            filename_root = filename.split(".")[0]
            
            # Find the contours box rectangle for each partido 
            cont_img, self.outputs = self.find_contour(c_image, i_path)
            
            for o in self.outputs:
                # Iterate over the rectangular boxes
                for k, v in o.items():
                    #print(f"WORKING ON {k}")
                    partido_key_id = k  
                    
                    p1, p2 = v
                    
                    # Draw Reactable for 
                    self.draw_rectangle_numpy(cont_img, p1,p2 )

                    if k in self.partidos_todos:
                        # Mueva Position tira 
                        n_p1 = (p1[0] + 340, p1[1])
                        n_p2 = (p2[0] + 180, p2[1])

                    if k in self.otros_todos:
                        # Mueva Position tira 
                        n_p1 = (p1[0] + 310, p1[1])
                        n_p2 = (p2[0] + 215, p2[1])

                    # Draw Reactable again.
                    self.draw_rectangle_numpy(cont_img, n_p1, n_p2 )
                    
                    # Letters as image |X|X|X|

                    # Obtain the numbers coordinates
                    x = n_p1[0]
                    y = n_p1[1]

                    w = n_p2[0]
                    h = n_p2[1]

                    # Cut this coordinates 
                    numbers_cout = cont_img[y:h, x:w]

                    # Find again the letters with the contour detector:
                    l_h, l_w, _ = numbers_cout.shape
                    
                    # Attempt to cut each letter
                    votes = []

                    # Iterate over this expected tree number images
                    for i in range(0,3):
                        
                        # Hard code possible position of numbers
                        
                        # Sliding x-window of 50 pixels
                        x0 = i*50       + 5
                        x1 = (i+1)*50   + 5

                        # New Points 
                        p1_l = (x0, 0)
                        p2_l = (x1, l_h)
                        
                        #####cv2.rectangle(numbers_cout, p1_l, p2_l, (244, 255, 0), 2)
                        
                        ####cv2.imwrite(f"actas/cuts/partidos/numbers/{filename_two}-{k}.jpg", numbers_cout)

                        
                        # Adjust each letter to his aprox box
                        # the labels for each latter are 0 , 1 and else
                        if (i == 0):
                            p1_l_n = (p1_l[0] - 5, p1_l[1])
                            p2_l_n = (p2_l[0], p2_l[1]-5)
                            letter = self.cut_image(numbers_cout, p1_l_n, p2_l_n, simple=True)
                        elif (i == 1):
                            p1_l_n = (p1_l[0]-5, p1_l[1])
                            p2_l_n = (p2_l[0]-45, p2_l[1]-1) #from  (p2_l[0]-50, p2_l[1]-3)
                            
                            letter = self.cut_image(numbers_cout, p1_l_n, p2_l_n,simple=True)
                        else:
                            p2_l_n = (p2_l[0]-120, p2_l[1]+10)
                            p2_l_n = (p2_l[0]+3, p2_l[1]+3)
                            letter = self.cut_image(numbers_cout, p1_l, p2_l_n, simple=True)

                        # PREDICT
                        #prediction = 1 #self.model.main_prediction(letter)

                        prediction = self.call_server_prediction(letter)

                        votes.append(prediction)
                        
                        if self.cut_numbers:
                            # Path where save the digit numbers
                            os.makedirs("results_votes/numbers/", exist_ok=True)

                            full_path = f"results_votes/numbers/{k}-{i}-{filename_root}.jpg"

                            # Save letter image
                            cv2.imwrite(full_path, letter)
                    
                    self.data_handler.append({partido_key_id: [votes, v]})       
                    
                    # IF needed, save the  |X|X|X| image
                    # os.makedirs("results/numbers_3XXX/", exist_ok=True)

                    #cv2.imwrite(f"results/numbers_3XXX/{filename_root}-{k}-.jpg", numbers_cout)
                    
                _file_name_log = "results_log"
                
                # Iterate over the results |X|X|X|
                # and join the results into one single integer
                for p in self.data_handler:
                
                    # Get the dict
                    for p_name, values in p.items():
                        partido_id_name = p_name
                        my_predictions, points = values

                        # join number predictions
                        my_vote = ""
                        for p in my_predictions:
                            if p == 10: # Blank prediction
                                p = "0"
                            my_vote = my_vote + str(p)

                        # Obtain the relative position with respect to the original image
                        p1, p2 = points

                        p_text = (p2[0]-15, p2[1])

                        # Draw Result
                        cv2.putText(cont_img, my_vote, tuple(p_text), self.font, 
                                                        self.fontScale,
                                                        self.fontColor,
                                                        self.lineType)
                        # Draw ID name
                        cv2.putText(cont_img, 
                                str(self.all_names_dict[partido_id_name]),
                                tuple([x-300,y+50]), 
                                self.font, 
                                self.fontScale,
                                self.fontColor,
                                self.lineType)

                        # Write to file the count
                        self.write_row_results_log(_file_name_log, 
                                                        filename,
                                                        my_vote,
                                                        partido_id_name)

                self.data_handler=[]

                # Write the acta with the number of results drawed on it.
            if self.draw_results:

                os.makedirs("results_votes/counts", exist_ok=True)
                cv2.imwrite(f"results_votes/counts/{filename}", cont_img)
                
            # except Exception as e:
            #     filename = i_path.split("/")[-1]
            #     _filename_error = "errorInOpenFile"
            #     self.write_row_debug_log(_filename_error, filename, e)
            #     print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Actas de mesa to counts')

    parser.add_argument('--data_path', type=str, default="../images/", help='Where is your jpg image data. absolute path')
    parser.add_argument('--draw_results', type=bool,default=True, help='Input dir for videos')
    
    
    args = parser.parse_args()

    images_path = args.data_path
    draw_results = args.draw_results
    
    # Images lists...
    images_list = glob.glob(f"{images_path}*.jpg")

    image_hanlder = VotesCounter( draw_results=True ) 

    image_hanlder.main(
        images_list
    )

    # RUN WITH VALIDATION Data WITH

    # python votes_counter.py --data_path=../test_dataset/validation/