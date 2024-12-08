#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
from shared import logging
from shared import constants as c
import logging as lg
from ph_strips import StripProcessor
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

method_captions = {'Square Center Point': c.METHOD_CENTER,
                   'Square Mean': c.METHOD_MEAN,
                   'Square Median': c.METHOD_MEDIAN,
                   'Square Median Inner 2/3': c.METHOD_INNER_MEDIAN,
                   'Square Median Outer 1/3': c.METHOD_OUTER_MEDIAN}


# FIX to remove the streamlit default handler and initialize our own handler
handlers = lg.getLogger().handlers
for handler in handlers:
    lg.getLogger().handlers.remove(handler)

logger = logging.Logger(name='test_handler',
                        handlers=[{'handler_class': logging.SCREEN_HANDLER,
                                   'formatter': logging.DEFAULT_FORMATTER,
                                   'level': 20}])


def save_outputs(output_path: str, entry_name: str, data: pd.DataFrame, img: np.array):
    """
    Store the detected values and image
    :param output_path:
    :param entry_name
    :param data:
    :param img:
    :return:
    """

    out_name_img = entry_name.split('.')[0] + '_processed.' + entry_name.split('.')[1]
    out_name_df = entry_name.split('.')[0] + '_processed.csv'

    cv2.imwrite(os.path.join(output_path, out_name_img),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    data.to_csv(os.path.join(output_path, out_name_df),
                index=False)


st.title('Ph Test strips processor')

st.markdown(
    """This tool is designed to automatically detect the results of multiple pH test strips. Please select the image 
    you want to process.
    """)


# c1, c2, c3 = st.columns(3)
# with c1:
#     preview = st.checkbox("Include preview", value=False)
# with c2:
#     automatic_calibration = st.checkbox("Calibrate automatically", value=False)
# with c3:
#     object_size_px = st.number_input(label='Object size in pixels:', min_value=0, max_value=10000, value=32)
# strip_type = st.selectbox(label='Type of strip:', options=('Macherey-Nagel pH-Fix 4.5-10.0', ''))
# number_of_strips = st.number_input(label='Number of pH strips in image:', min_value=1, max_value=10, value=3)
# detections_per_strip = st.number_input(label='Number of detections per pH:', min_value=1, max_value=10, value=3)

with st.sidebar:
    preview = st.checkbox("Include preview", value=False)
    store_output = st.checkbox("Store output", value=False)
    if store_output:
        output_path = st.text_input(label='Output path:', value=os.getcwd())
    restrict_viewpoint = st.checkbox("Restrict viewpoint",
                                     value=True,
                                     help='Restricts detections only to the square defined by Aruco markers')
    # automatically calculate object size in pixels based on aruco marker distance
    automatic_calibration = st.checkbox("Calibrate automatically",
                                        value=True,
                                        help='Automatically calculate square size in pixels based on aruco markers')
    if not automatic_calibration:
        object_size_px = st.number_input(label='Object size in pixels:', min_value=0, max_value=10000, value=32,
                                         disabled=automatic_calibration)
        # only allow to rotate image when we are using the automatic detection
        rotate_image = False
    else:
        object_size_px = None
        # rotate the image to align strips vertically
        rotate_image = st.checkbox("Align vertically",
                                   value=True,
                                   help='Rotate the image to align strips vertically')

    calculation_method = st.selectbox(label='Select the calculation method', options=list(method_captions.keys()))
    strip_type = st.selectbox(label='Type of strip:', options=('Macherey-Nagel pH-Fix 4.5-10.0', ''))
    number_of_strips = st.number_input(label='Number of pH strips in image:', min_value=1, max_value=10, value=3)
    detections_per_strip = st.number_input(label='Number of detections per pH:', min_value=1, max_value=10, value=3)
    crop_margin = st.number_input(label='Margin of the image to crop:', min_value=0, max_value=100, value=0,
                                  help='Crop sides of the image when viewpoint restriction is not selected')

uploaded_file = st.file_uploader(label='Upload a file to process', accept_multiple_files=False)
if uploaded_file is not None:
    image = Image.open(uploaded_file)  # .transpose(Image.ROTATE_270)
    # change orientation based on EXIF orientation tag
    image = ImageOps.exif_transpose(image)
    img_array = np.array(image)

    if crop_margin > 0:
        margin_px = int(crop_margin/100 * min(img_array.shape[0], img_array.shape[1]))
        img_array = img_array[margin_px:-margin_px, margin_px:-margin_px,:]

    # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    if preview:
        st.caption(f'Preview of file: {uploaded_file.name}')
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        ax_p.imshow(image)
        ax_p.set_axis_off()
        st.pyplot(fig_p)

    count, out_img, df = StripProcessor(logger=logger, number_of_strips=number_of_strips,
                                        check_object_count=True,
                                        calculation_method=method_captions[calculation_method]).process_image(
        image_path='',
        output_folder='',
        detect_size=automatic_calibration,
        detected_object_size=object_size_px,
        image=img_array,
        store_results=False,
        rotate_image=rotate_image,
        restrict_viewpoint=restrict_viewpoint)

    st.caption(f'Displaying file: {uploaded_file.name}')

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(out_img)
    ax.set_axis_off()
    st.pyplot(fig)

    st.caption('Detected objects')

    st.dataframe(data=df, hide_index=True) #, use_container_width=st.session_state.use_container_width)

    if store_output:
        if st.button('Save results'):
            save_outputs(output_path=output_path,
                         entry_name=uploaded_file.name,
                         data=df,
                         img=out_img)
