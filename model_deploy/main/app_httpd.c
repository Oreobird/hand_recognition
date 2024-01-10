
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "nvs_flash.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_wifi.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "app_common.h"
#include "app_camera.h"


#define APP_WIFI_SSID        "test2_4"
#define APP_WIFI_PWD         "12345678"

#define PART_BOUNDARY "123456789000000000000987654321"

static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %d.%06d\r\n\r\n";

static esp_netif_t *s_netif_sta = NULL;
static httpd_handle_t s_httpd = NULL;

static int app_camera_get_jpg_data(cam_jpg_data_t *jpg_data)
{
    if (jpg_data == NULL)
    {
        return APP_FAIL;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb)
    {
        printf("Camera capture failed\n");
        return APP_FAIL;
    }

    jpg_data->ts.tv_sec = fb->timestamp.tv_sec;
    jpg_data->ts.tv_usec = fb->timestamp.tv_usec;

    bool jpeg_converted = frame2jpg(fb, 8, &jpg_data->data, (unsigned int*)&jpg_data->data_len);
    esp_camera_fb_return(fb);
    fb = NULL;
    if (!jpeg_converted)
    {
        printf("JPEG compression failed\n");
        return APP_FAIL;
    }
    return APP_OK;
}


static esp_err_t stream_handler(httpd_req_t *req)
{
    int ret = APP_OK;
    char *part_buf[128] = {0};
    cam_jpg_data_t jpg_data = {0};

    ret = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (ret != ESP_OK)
    {
        return ret;
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "60");

    while (true)
    {
        ret = app_camera_get_jpg_data(&jpg_data);
        if (ret == APP_OK)
        {
            ret = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        if (ret == APP_OK)
        {
            size_t hlen = snprintf((char *)part_buf, 128, _STREAM_PART, jpg_data.data_len, jpg_data.ts.tv_sec, jpg_data.ts.tv_usec);
            ret = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        }
        if (ret == APP_OK)
        {
            ret = httpd_resp_send_chunk(req, (const char *)jpg_data.data, jpg_data.data_len);
        }

        if (jpg_data.data)
        {
            free(jpg_data.data);
        }

        jpg_data.data = NULL;

        if (ret != APP_OK)
        {
            break;
        }
    }

    return ret;
}


httpd_uri_t uri_handler_jpg = {
    .uri = "/",
    .method = HTTP_GET,
    .handler = stream_handler};

static int start_webserver(void)
{
    if (s_httpd != NULL)
    {
        return APP_OK;
    }

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    // Start the httpd server
    printf("Starting server on port: '%d'\n", config.server_port);
    esp_err_t ret = httpd_start(&s_httpd, &config);
    if (ret == ESP_OK)
    {
        // Set URI handlers
        printf("Registering URI handlers\n");
        httpd_register_uri_handler(s_httpd, &uri_handler_jpg);
        return APP_OK;
    }

    printf("Error starting server: 0x%x\n", ret);
    return APP_FAIL;
}

static void stop_webserver(void)
{
    if (s_httpd)
    {
        httpd_stop(s_httpd);
        s_httpd = NULL;
    }
}


static void wifi_event_hdl(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT)
    {
        switch (event_id)
        {
            case WIFI_EVENT_STA_DISCONNECTED:
                stop_webserver();
                break;
            default:
                break;
        }
    }
    else if (event_base == IP_EVENT)
    {
        switch (event_id)
        {
            case IP_EVENT_STA_GOT_IP:
            {
                char sta_ip_str[16] = {0};
                esp_netif_ip_info_t sta_ip;
                if (s_netif_sta && ESP_OK == esp_netif_get_ip_info(s_netif_sta, &sta_ip))
                {
                    esp_ip4addr_ntoa(&sta_ip.ip, sta_ip_str, 16);
                    printf("Device IP: %s \n", sta_ip_str);
                    start_webserver();
                }
                break;
            }
            default:
                break;
        }
    }
}


static int app_wifi_init(void)
{
    nvs_flash_init();
    esp_event_handler_instance_t inst_any_id;
    esp_event_handler_instance_t inst_got_ip;
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                        ESP_EVENT_ANY_ID,
                                        &wifi_event_hdl,
                                        NULL,
                                        &inst_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                        ESP_EVENT_ANY_ID,
                                        &wifi_event_hdl,
                                        NULL,
                                        &inst_got_ip));

    if (NULL == s_netif_sta)
    {
        s_netif_sta = esp_netif_create_default_wifi_sta();
        if (!s_netif_sta)
        {
            printf("create sta fail\n");
            return APP_FAIL;
        }
    }

    ESP_ERROR_CHECK(esp_wifi_start());

    wifi_config_t wifi_config;
    memset(&wifi_config, 0, sizeof(wifi_config_t));
    snprintf((char*)wifi_config.sta.ssid, 32, "%s", APP_WIFI_SSID);
    snprintf((char*)wifi_config.sta.password, 64, "%s", APP_WIFI_PWD);

    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));

    esp_wifi_connect();
    return APP_OK;
}


void app_httpd_init(void)
{
    app_wifi_init();
}