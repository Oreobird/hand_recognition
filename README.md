# Hand Recognition
In these project, we use TensorFlow to train a gesture recognition model in Colab environment. Then Deploy the model to ESP32 through the ESP-DL deep learning component, and perform real-time gesture recognition by collecting camera data.
## model_develop
Run train_hand_rec.ipynb in Google Colab for model training, will call the following scripts for training and quantization
* prepare_data.py: Prepare training/testing and calibration data.
* train.py: Create model and train model.
* esp_dl_quant.py: Use esp-dl quantization tools to quantize model

After training and quantization, we can get 3 files in trained_model:
* hand_rec_mode_coefficient.cpp
* hand_rec_model_coefficient.hpp
* hand_rec_model_model.hpp

Download these files to directory model_deploy/main/

## model_deploy
* Download ESP-IDF SDK (>= v5.0 is recommended) and setup the env

```
cd esp-idf_v5.0
./install.sh
. ./export.sh
```
* Download project source code
```
git clone --depth=1 --recursive https://github.com/Oreobird/hand_recognition.git
```
* Compile and run project

```
cd model_deploy
idf.py set-target esp32
idf.py flash -b 921600 -p /dev/ttyUSB0 monitor
```
