
#include <stdlib.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "app_common.h"
#include "app_camera.h"
#include "app_httpd.h"

#include "dl_tool.hpp"
#include "hand_rec_model_model.hpp"


void app_task_entry(void *arg)
{
    int ret = app_camera_init();
    if (ret != APP_OK)
    {
        printf("camera init fail\n");
        return;
    }

    app_httpd_init();

    int img_size = 96 * 96;
    int8_t *in_data = (int8_t *)dl::tool::malloc_aligned_prefer(img_size, sizeof(int8_t), 16);
    if (in_data == NULL)
    {
        printf("malloc fail\n");
        return;
    }

    HAND_REC_MODEL model;
    model.input(in_data);

    while (true)
    {
#ifdef STATIC_IMG_TEST
        app_camera_get_raw_data(s_raw_data, img_size, in_data);
#else
        app_camera_get_raw_data(NULL, img_size, in_data);
#endif
        dl::tool::Latency latency;
        latency.start();
        float *score = model.invoke();
        latency.end();
        latency.print("\nSIGN", "forward");

        float max_score = score[0];
        int max_index = 0;
        for (size_t i = 0; i < 6; i++)
        {
            printf("%f, ", score[i]*100);
            if (score[i] > max_score)
            {
                max_score = score[i];
                max_index = i;
            }
        }
        printf("\n");

        switch (max_index)
        {
            case 0:
                printf("Palm: 0\n");
                break;
            case 1:
                printf("I: 1\n");
                break;
            case 2:
                printf("Thumb: 2\n");
                break;
            case 3:
                printf("ok: 3\n");
                break;
            case 4:
                printf("C: 4\n");
                break;
            case 5:
                printf("background: 5\n");
                break;
            default:
                printf("No result\n");
                break;
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}


extern "C" void app_main()
{
    BaseType_t ret = xTaskCreatePinnedToCore(app_task_entry,
                                            "hand_rec",
                                            4096 / sizeof(portSTACK_TYPE),
                                            NULL,
                                            7,
                                            NULL,
                                            1);
    if (ret != pdPASS)
    {
        return;
    }
}
