#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from statistics import mean
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Union
import numpy as np
import os
from shared import logging
from shared import constants as c
from datetime import datetime


class StripProcessor(object):
    """
    Load either a single image containing multiple strips for ph detection or all images within a given folder. For each
    image detect all the ph squares (3 within each strip)
    """
    def __init__(self, logger, calculation_method: str,
                 number_of_strips: int = 3, check_object_count: bool = True):
        """

        :param logger:
        :param number_of_strips:
        :param check_object_count: check if the correct number of objects has been detected (number_of_strips*3)
        """
        self.logger = logger
        self.calculation_method = calculation_method
        self.number_of_strips = number_of_strips
        self.check_object_count = check_object_count

    @staticmethod
    def get_contour_precedence(contour: np.array, height: int, tolerance_factor: int = 35) -> int:
        """
        Function to order the contours top-to-bottom and left-to-right. We need to add a tolerance, because the image
        can be tilted or the slips can be at slightly different heights. It is more stable to do it top to bottom,
        as the distance in that dimension stays the same.
        # https://stackoverflow.com/questions/39403183/python-opencv-sorting-contours
        :param contour:
        :param height:
        :param tolerance_factor
        :return:
        """
        x, y, _, _ = cv2.boundingRect(contour)
        # return ((y // tolerance_factor) * tolerance_factor) * cols + x
        # transformation to add together values from x and y axis to give a unique order for each object
        return ((x // tolerance_factor) * tolerance_factor) * height + y

    def detect_squares_in_image(self,
                                image: np.array,
                                mask: np.array,
                                obj_size: int,
                                obj_tolerance: float) -> Tuple[np.array, pd.DataFrame, int]:
        """
        Based on the mask with the thresholded image find all contours of square like images with a given shape and
        within a given object area. Mark the position on the image for visual inspection and store the position
        and colour of the object in a dataframe.

        :param image: original image
        :param mask: mask based on colour threshold
        :param obj_size: object size to search for
        :param obj_tolerance: tolerance object size to search for
        :return:
        """
        self.logger.info(f'Detecting ph squares in the image, image dimensions: {image.shape}, with mask {mask.shape}')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # blur for color selection of the center of the image
        blur = cv2.medianBlur(image, 9)

        # find the contours in the image
        cnts = cv2.findContours(255 - morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_SIMPLE)

        # Modification so that it works with different versions of OpenCV (len(cnts) == 2 is v2.4)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # create output object
        output = []
        # sorting top-to-bottom and then left-to-right
        sorted_cnts = sorted(cnts, key=lambda ctr: self.get_contour_precedence(ctr, image.shape[0],
                                                                               math.floor(obj_size/2)))

        count = 0
        # restrict to only contours of interest
        for sc in sorted_cnts:
            area = cv2.contourArea(sc)
            x, y, w, h = cv2.boundingRect(sc)
            ratio = w / h
            # detect only square like objects with an area within a given range
            if .7 < ratio < 1.4 and (obj_size * (1 - obj_tolerance)) ** 2 < area < (
                   obj_size * (1 + obj_tolerance)) ** 2:
                # print(x,y,w,h)
                count += 1
                # add description and marker of the created object
                cv2.putText(image, str(count), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, obj_size/30,
                            (0, 0, 0), math.floor(obj_size/30))
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                # print('object id:', count, 'coordinates:', (x, y), 'object color(R,G,B):',
                #      blur[int(y + h / 2), int(x + w / 2), :], ' area:', area, ' ratio:', ratio)

                self.logger.info(f'Detected object coordinates: x1={x}, x2={x+w}, y1={y}, y2={y+h}')
                # calculate the rgb colours
                rgb = self.extract_rgb(image=blur[y:y+h, x:x+w, :])

                self.logger.debug(f'id: {count} coordinates: {x} {y}, color(R,G,B): {rgb}')
                self.logger.debug(f'id: {count} area: {area} ratio: {ratio}')

                # store the ID, center point of the object and RGB color
                output.append([count, x, y, rgb[0], rgb[1], rgb[2]])

        # plt.imshow(image, cmap='gray')

        self.logger.info(f'Found: {len(output)} ph squares in the image of the required size')

        df = pd.DataFrame(np.array(output),
                          columns=['id', 'x_coordinate', 'y_coordinate', 'R', 'G', 'B'])
        # add a timestamp
        df['timestamp'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        # add calculation method
        df['method'] = self.calculation_method

        # return the marked image and the statistics
        return image, df, count

    def extract_rgb(self, image) -> Tuple[float, float, float]:
        """
        Calculate the RGB colours based on a provided calculation method
        :param image:
        :return:
        """
        assert self.calculation_method in [c.METHOD_MEAN, c.METHOD_CENTER, c.METHOD_MEDIAN,
                                           c.METHOD_INNER_MEDIAN,
                                           c.METHOD_OUTER_MEDIAN], 'Calculation method not supported'
        self.logger.info(f"Image shape: {image.shape}")
        if self.calculation_method == c.METHOD_MEDIAN:
            r, g, b = np.median(image[:, :, 0]), np.median(image[:, :, 1]), np.median(image[:, :, 2])
        elif self.calculation_method == c.METHOD_MEAN:
            r, g, b = np.mean(image[:, :, 0]), np.mean(image[:, :, 1]), np.mean(image[:, :, 2])
        elif self.calculation_method == c.METHOD_CENTER:
            r, g, b = image[image.shape[0] // 2, image.shape[1] // 2, :]
        elif self.calculation_method == c.METHOD_INNER_MEDIAN:
            # calculate one third distance
            w = image.shape[0] // 3
            h = image.shape[1] // 3
            center_x = image.shape[0] // 2
            center_y = image.shape[1] // 2
            r = np.median(image[center_x-w:center_x+w, center_y-h:center_y+h, 0])
            g = np.median(image[center_x-w:center_x+w, center_y-h:center_y+h, 1])
            b = np.median(image[center_x-w:center_x+w, center_y-h:center_y+h, 2])
        elif self.calculation_method == c.METHOD_OUTER_MEDIAN:
            # calculate one third distance
            w = image.shape[0] // 3
            h = image.shape[1] // 3
            center_x = image.shape[0] // 2
            center_y = image.shape[1] // 2
            mask = np.zeros(image.shape, 'bool')
            # mask the inner third of the image
            mask[center_x-w:center_x+w, center_y-h:center_y+h, :] = 1
            # create a masked array
            masked_image = np.ma.masked_array(image, mask=mask)
            # calculate the median values in masked array, use ma.median else the mask is ignored
            r = np.ma.median(masked_image[:, :, 0])
            g = np.ma.median(masked_image[:, :, 1])
            b = np.ma.median(masked_image[:, :, 2])

        return r, g, b

    def restrict_hsv_range(self,
                           image: np.array, min_hsv: tuple, max_hsv: tuple) -> np.array:
        """
        Restrict the colours we want based on HSV values within a given range
        :param image:
        :param min_hsv: min values for hsv
        :param max_hsv: max values for hsv
        :return:
        """
        self.logger.debug('Restricting to hsv colour range')
        # blur image
        blur = cv2.medianBlur(image, 7)
        # convert to hsv to apply colour restriction
        hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
        # restrict the image to a given hsv range
        return cv2.inRange(hsv, min_hsv, max_hsv)

    def get_distance_between_markers(self,
                                     marker_ids: np.array,
                                     marker_corners: tuple,
                                     id1: int,
                                     id2: int) -> Union[float, None]:
        """
        Calculate the distance between to markers based on their ids
        :param marker_ids:
        :param marker_corners:
        :param id1:
        :param id2:
        :return:
        """
        # find the position of the marker based on id
        for i in range(0, marker_ids.shape[0]):
            # if marker found extract center point
            if marker_ids[i][0] == id1:
                c1 = marker_corners[i][0]
                x1 = c1[:, 0].mean()
                y1 = c1[:, 1].mean()
            elif marker_ids[i][0] == id2:
                c2 = marker_corners[i][0]
                x2 = c2[:, 0].mean()
                y2 = c2[:, 1].mean()
        try:
            # calculate the diagonal based on the x and y coordinates of the 2 markers
            return (abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5
        # when any of the local variables is not declared return None meaning that distance could not be calculated
        # because one of the ids was not found
        except UnboundLocalError:
            self.logger.info(f'Unable to find distance for marker pair {id1} and {id2}')
            return np.nan
        except Exception as e:
            print(e)
            raise

    def detect_markers(self,
                       image_path: str,
                       image: Union[np.array, None] = None) -> Tuple[float, float]:

        self.logger.info('Detecting marker shape in image to determine the size of shapes to detect')
        if image is None:
            # load image
            image = cv2.imread(image_path)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        marker_corners, marker_ids, _ = detector.detectMarkers(image)

        self.logger.info(f'Detected {len(marker_ids)} Aruco markers in the image')

        # calculate the various horizontal and vertical distances between markers
        vertical_distance_1 = c.VERTICAL_DISTANCE / self.get_distance_between_markers(marker_ids=marker_ids,
                                                                                      marker_corners=marker_corners,
                                                                                      id1=c.TOP_LEFT_MARKER_ID,
                                                                                      id2=c.BOTTOM_LEFT_MARKER_ID)

        vertical_distance_2 = c.VERTICAL_DISTANCE / self.get_distance_between_markers(marker_ids=marker_ids,
                                                                                      marker_corners=marker_corners,
                                                                                      id1=c.TOP_RIGHT_MARKER_ID,
                                                                                      id2=c.BOTTOM_RIGHT_MARKER_ID)

        horizontal_distance_1 = c.HORIZONTAL_DISTANCE / self.get_distance_between_markers(marker_ids=marker_ids,
                                                                                          marker_corners=marker_corners,
                                                                                          id1=c.TOP_LEFT_MARKER_ID,
                                                                                          id2=c.TOP_RIGHT_MARKER_ID)

        horizontal_distance_2 = c.HORIZONTAL_DISTANCE / self.get_distance_between_markers(marker_ids=marker_ids,
                                                                                          marker_corners=marker_corners,
                                                                                          id1=c.BOTTOM_LEFT_MARKER_ID,
                                                                                          id2=c.BOTTOM_RIGHT_MARKER_ID)
        horizontal_angle_1 = self.find_rotation_angle(marker_ids=marker_ids,
                                                      marker_corners=marker_corners,
                                                      id1=c.TOP_LEFT_MARKER_ID,
                                                      id2=c.BOTTOM_LEFT_MARKER_ID)

        horizontal_angle_2 = self.find_rotation_angle(marker_ids=marker_ids,
                                                      marker_corners=marker_corners,
                                                      id1=c.TOP_RIGHT_MARKER_ID,
                                                      id2=c.BOTTOM_RIGHT_MARKER_ID)

        rotation_angle = - float(np.nanmean([horizontal_angle_1, horizontal_angle_2]))

        self.logger.info(f'Current angle deviation from vertical position {rotation_angle} degrees')

        # ignore markers we haven't found = np.nan
        return float(np.nanmean([horizontal_distance_1, horizontal_distance_2,
                                 vertical_distance_1, vertical_distance_2])), rotation_angle

    def find_rotation_angle(self, marker_ids, marker_corners, id1, id2):
        """
        Find the angle that the image needs to be rotated by to achieve that the strips are vertically aligned
        :param marker_ids:
        :param marker_corners:
        :param id1:
        :param id2:
        :return:
        """
        # find the position of the marker based on id
        for i in range(0, marker_ids.shape[0]):
            # if marker found extract center point
            if marker_ids[i][0] == id1:
                c1 = marker_corners[i][0]
                x1 = c1[:, 0].mean()
                y1 = c1[:, 1].mean()
            elif marker_ids[i][0] == id2:
                c2 = marker_corners[i][0]
                x2 = c2[:, 0].mean()
                y2 = c2[:, 1].mean()
        try:
            # calculate the current angle between markers that should be vertical
            return math.degrees(math.atan((x1 - x2) / (y1 - y2)))
        # when any of the local variables is not declared return None meaning that distance could not be calculated
        # because one of the ids was not found
        except UnboundLocalError:
            self.logger.info(f'Unable to find distance for marker pair {id1} and {id2}')
            return np.nan
        except Exception as e:
            print(e)
            raise

    def restrict_viewpoint(self,
                           image_path: str,
                           image: Union[np.array, None] = None) -> np.array:
        """
        Based on aruco markers detect a mask of the viewpoint of interest
        :param image_path:
        :param image:
        :return:
        """

        self.logger.info('Detecting viewpoint of interest within the image')
        if image is None:
            # load image
            image = cv2.imread(image_path)

        # define which dictionary to use
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        # use default parameters
        parameters = cv2.aruco.DetectorParameters()
        # initialize detector
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        # run detections
        marker_corners, marker_ids, _ = detector.detectMarkers(image)

        # initialize the variable to the maximum values they can be
        y_min = image.shape[0]
        y_max = 0
        x_min = image.shape[1]
        x_max = 0

        marker_sides = []
        # across all objects find the most outer points in each dimension x, y (min and max)
        for i in range(0, len(marker_ids)):
            # get the size of the marker square side
            marker_sides.append(marker_corners[i][0][:, 1].max() - marker_corners[i][0][:, 1].min())
            if marker_corners[i][0][:, 1].min() < y_min:
                y_min = marker_corners[i][0][:, 1].min()
            if marker_corners[i][0][:, 1].max() > y_max:
                y_max = marker_corners[i][0][:, 1].max()
            if marker_corners[i][0][:, 0].min() < x_min:
                x_min = marker_corners[i][0][:, 0].min()
            if marker_corners[i][0][:, 0].max() > x_max:
                x_max = marker_corners[i][0][:, 0].max()

        marker_side = int(mean(marker_sides))

        self.logger.info(f'Viewpoint restricted to x in ({x_min}, {x_max}) and y in ({y_min}, {y_max})')

        # create a mask of the same dimensions as the image
        mask = np.zeros(image.shape[:2], dtype='uint8')

        # apply restrictions to mask, setting all points outside our range to zero
        mask[int(y_min+marker_side):int(y_max-marker_side), int(x_min+marker_side):int(x_max-marker_side), ] = 1

        return mask

    def process_image(self,
                      image_path: str,
                      output_folder: str,
                      detect_size: bool = True,
                      detected_object_size: int = c.DEFAULT_OBJECT_SIZE,
                      image: Union[np.array, None] = None,
                      store_results: bool = True,
                      restrict_viewpoint: bool = True,
                      rotate_image: bool = True,
                      debug: bool = False):
        """
        Detect the marker object in the image. Find all objects within a defined colour range. For that colour range run


        :param image_path:
        :param output_folder:
        :param detect_size:
        :param detected_object_size:
        :param image:
        :param store_results:
        :param restrict_viewpoint:
        :param rotate_image:
        :param debug:
        :return:
        """
        self.logger.info(f'Object detection started for image: {image_path}')
        if detect_size:
            self.logger.info('Detecting marker object distance')

            average_distance, rotation_angle = self.detect_markers(image_path=image_path,
                                                                   image=image)

            # check that the distance is not NAN, else replace with default value
            detected_object_size = c.REAL_OBJECT_SIZE / average_distance if not np.isnan(
                average_distance) else detected_object_size

            self.logger.info(f'Detected object size: {detected_object_size} in pixels')
        else:
            detected_object_size = detected_object_size

        if image is None:
            # load image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if rotate_image and detect_size:
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, rotation_angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        # apply to different colour filters for different colour ranges
        mask1 = self.restrict_hsv_range(image=image, min_hsv=c.MIN_HSV_1, max_hsv=c.MAX_HSV_1)
        mask2 = self.restrict_hsv_range(image=image, min_hsv=c.MIN_HSV_2, max_hsv=c.MAX_HSV_2)
        # create a unified mask for both hsv colour ranges
        mask = cv2.bitwise_or(mask1, mask2)

        # smooth mask
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 5)

        if restrict_viewpoint:
            # find the center are between aruco markers and return mask
            mask_restr = self.restrict_viewpoint(image_path=image_path, image=image)
            # multiply 2 matrices, all pixels outside the viewpoint are multiplied by 0
            mask = mask * mask_restr

        if store_results and debug:
            self.logger.info(f'Storing mask for image: {image_path}')
            file_name = os.path.normpath(image_path).split(os.path.sep)[-1]

            out_mask_img = file_name.split('.')[0] + '_mask.' + file_name.split('.')[1]

            cv2.imwrite(os.path.join(output_folder, out_mask_img),
                        cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

        img, df, count = self.detect_squares_in_image(image=image,
                                                      mask=mask,
                                                      obj_size=detected_object_size,
                                                      obj_tolerance=0.4)

        # debug functionality to show visualize viewpoint
        if debug:
           img = img*np.stack((mask,)*3, axis=-1)

        # check that the correct number of objects was detected
        if self.check_object_count:
            if count != self.number_of_strips * 3:
                self.logger.warning(
                    'Incorrect number of objects detected, validate manually! Expected={0}, Detected={1}'.format(
                        self.number_of_strips * 3,
                        count
                    ))

        #
        if store_results:
            self.logger.info(f'Storing output for image: {image_path}')
            file_name = os.path.normpath(image_path).split(os.path.sep)[-1]

            out_name_img = file_name.split('.')[0] + '_processed.' + file_name.split('.')[1]
            out_name_df = file_name.split('.')[0] + '_processed.csv'

            cv2.imwrite(os.path.join(output_folder, out_name_img),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            df.to_csv(os.path.join(output_folder, out_name_df),
                      index=False)

        return count, img, df

    def bulk_process_images(self,
                            input_folder: str,
                            output_folder: str):
        """
        Process all files within a folder

        :param input_folder:
        :param output_folder:
        :return:
        """
        total = 0
        files_processed = 0
        all_files = sorted(os.scandir(input_folder), key=lambda e: e.name)
        for entry in all_files:
            if entry.name.endswith(".jpeg") and entry.is_file():
                self.logger.info('Processing file: {0}'.format(entry.name))
                count, _, _ = self.process_image(image_path=os.path.join(input_folder, entry.name),
                                                 output_folder=output_folder,
                                                 detect_size=True)
                total += count
                files_processed += 1
        self.logger.info(f'process finished. Found {total} objects in {files_processed} files')


if __name__ == '__main__':

    logger = logging.Logger(name='test_handler',
                            handlers=[{'handler_class': logging.SCREEN_HANDLER,
                                       'formatter': logging.DEFAULT_FORMATTER,
                                       'level': 20}])

    IMAGE_PATH = '/Volumes/Macintosh HD/Users/matt/wrk/ffuk/ondra_prouzky/data/aruco/IMG_0538.jpeg'

    # run single detection
    StripProcessor(logger=logger, calculation_method=c.METHOD_MEDIAN).process_image(
        image_path=IMAGE_PATH,
        output_folder='/Users/matt/wrk/ffuk/ondra_prouzky/',
        detect_size=True,
        restrict_viewpoint=False,
        debug=True)
    # run bulk detection
    #StripProcessor(logger=logger, calculation_method=c.METHOD_MEDIAN).bulk_process_images(
    #    input_folder='/Volumes/Macintosh HD/Users/matt/wrk/ffuk/ondra_prouzky/data/aruco/',
    #    output_folder='/Volumes/Macintosh HD/Users/matt/wrk/ffuk/ondra_prouzky/data/aruco/out/')
