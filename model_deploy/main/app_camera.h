#ifndef __APP_CAMERA_H__
#define __APP_CAMERA_H__


#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "esp_camera.h"

typedef struct
{
    struct timeval ts;
    uint32_t data_len;
    uint8_t *data;
} cam_jpg_data_t;

int app_camera_init();

int app_camera_get_raw_data(uint8_t *src_data, int img_size, int8_t* img_data);

#ifdef __cplusplus
}
#endif

#endif
