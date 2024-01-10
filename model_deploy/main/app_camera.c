
#include "sensor.h"
#include <stdbool.h>
#include "app_camera.h"
#include "app_common.h"

/**
 * PIXFORMAT_RGB565,    // 2BPP/RGB565
 * PIXFORMAT_YUV422,    // 2BPP/YUV422
 * PIXFORMAT_GRAYSCALE, // 1BPP/GRAYSCALE
 * PIXFORMAT_JPEG,      // JPEG/COMPRESSED
 * PIXFORMAT_RGB888,    // 3BPP/RGB888
 */
#define CAM_PIXEL_FORMAT PIXFORMAT_GRAYSCALE

/*
 * FRAMESIZE_96X96,    // 96x96
 * FRAMESIZE_QQVGA,    // 160x120
 * FRAMESIZE_QQVGA2,   // 128x160
 * FRAMESIZE_QCIF,     // 176x144
 * FRAMESIZE_HQVGA,    // 240x176
 * FRAMESIZE_QVGA,     // 320x240
 * FRAMESIZE_CIF,      // 400x296
 * FRAMESIZE_VGA,      // 640x480
 * FRAMESIZE_SVGA,     // 800x600
 * FRAMESIZE_XGA,      // 1024x768
 * FRAMESIZE_SXGA,     // 1280x1024
 * FRAMESIZE_UXGA,     // 1600x1200
 */
#define CAM_FRAME_SIZE FRAMESIZE_96X96

// #define CAM_PIN_SIOD 18
// #define CAM_PIN_SIOC 23
// #define CAM_PIN_VSYNC 5
// #define CAM_PIN_HREF 27
// #define CAM_PIN_XCLK 4
// #define CAM_PIN_PCLK 25
// #define CAM_PIN_D7 36
// #define CAM_PIN_D6 19
// #define CAM_PIN_D5 21
// #define CAM_PIN_D4 39
// #define CAM_PIN_D3 35
// #define CAM_PIN_D2 14
// #define CAM_PIN_D1 13
// #define CAM_PIN_D0 34
// #define CAM_PIN_PWDN 32
// #define CAM_PIN_RESET 33

#define CAM_PIN_PWDN 32
#define CAM_PIN_RESET -1 //software reset will be performed
#define CAM_PIN_XCLK 0
#define CAM_PIN_SIOD 26
#define CAM_PIN_SIOC 27

#define CAM_PIN_D7 35
#define CAM_PIN_D6 34
#define CAM_PIN_D5 39
#define CAM_PIN_D4 36
#define CAM_PIN_D3 21
#define CAM_PIN_D2 19
#define CAM_PIN_D1 18
#define CAM_PIN_D0 5
#define CAM_PIN_VSYNC 25
#define CAM_PIN_HREF 23
#define CAM_PIN_PCLK 22

#define XCLK_FREQ_HZ 20000000

int app_camera_init()
{
    /* IO13, IO14 is designed for JTAG by default,
    * to use it as generalized input,
    * firstly declare it as pullup input */
#if 0
    gpio_config_t conf;
    conf.mode = GPIO_MODE_INPUT;
    conf.pull_up_en = GPIO_PULLUP_ENABLE;
    conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    conf.intr_type = GPIO_INTR_DISABLE;
    conf.pin_bit_mask = 1LL << CAM_PIN_D0;
    gpio_config(&conf);
    conf.pin_bit_mask = 1LL << CAM_PIN_D1;
    gpio_config(&conf);
#endif
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.xclk_freq_hz = XCLK_FREQ_HZ;
    config.pixel_format = CAM_PIXEL_FORMAT;
    config.frame_size = CAM_FRAME_SIZE;
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_DRAM;
    // config.fb_location =  CAMERA_FB_IN_PSRAM;

    // camera init
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
    {
        printf("Camera init failed with error 0x%x\n", err);
        return APP_FAIL;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s == NULL)
    {
        printf("sensor get fail\n");
        return APP_FAIL;
    }

    if (s->id.PID == OV3660_PID)
    {
        s->set_vflip(s, 1);//flip it back
        s->set_brightness(s, 1);//up the blightness just a bit
        s->set_saturation(s, -2);//lower the saturation
    }

    printf("Camera sensor PID=%d", s->id.PID);
    return APP_OK;
}

int app_camera_get_raw_data(uint8_t *src_data, int img_size, int8_t* img_data)
{
    if (img_data == NULL)
    {
        return APP_FAIL;
    }

    if (src_data == NULL)
    {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb)
        {
            printf("Camera capture failed");
            return APP_FAIL;
        }

        for (int i = 0; i < img_size; i++)
        {
            img_data[i] = ((uint8_t *) fb->buf)[i] ^ 0x80;
        }

        esp_camera_fb_return(fb);
    }
    else
    {
        for (int i = 0; i < img_size; i++)
        {
            img_data[i] = ((uint8_t *) src_data)[i] ^ 0x80;
        }
    }
    return APP_OK;
}