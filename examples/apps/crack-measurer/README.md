This application measures the width and length of a crack on a concrete surface. To use
the application, you must first calibrate the camera by selecting a line on the image
and entering the actual distance of the line in inches. Once this is done, you can upload
an image with a crack and the crack will be segmented by a LandingLens model, and then
analyzed so the length and maximum width of the crack can be found.

To run this app, first install the requirements: `pip install -r examples/apps/crack-measurer/requirements.txt`

Then run: `streamlit run examples/apps/crack-measurer/app.py`
